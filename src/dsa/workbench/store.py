"""Durable project annotations and process jobs; never rewrite source datasets."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import uuid

from dsa.measure.semantic_phases import PHASES
from dsa.measure.semantic_pilot import sha
from dsa.pages.semantic_clip_labeling import annotation_registry, validate_export


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def progress_hash(progress):
    return fingerprint(progress['tasks'])


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, data):
        self.data = Path(data).resolve()
        self.directory = self.data/'.workbench'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory/'state.sqlite3'
        with self.connect() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS annotations (
                    project TEXT PRIMARY KEY, queue_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    payload TEXT NOT NULL, saved_at TEXT NOT NULL, progress_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, project TEXT NOT NULL, action TEXT NOT NULL,
                    status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    pid INTEGER, request TEXT NOT NULL, result TEXT, error TEXT);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def project(self, project):
        if not isinstance(project, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', project):
            raise ValueError('Invalid project ID')
        root = (self.data/'projects'/project).resolve()
        if not root.is_relative_to(self.data/'projects') or not (root/'manifest.json').is_file():
            raise ValueError('Project not found')
        return root

    def current(self, project):
        return json.loads((self.project(project)/'current-semantic.json').read_text())

    def queue(self, project):
        root = self.project(project)
        current = self.current(project)
        editor = self.inside(root, current['editor'])
        return json.loads((editor/'queue.json').read_text())

    @staticmethod
    def inside(root, relative):
        p = (root/relative).resolve()
        if Path(relative).is_absolute() or not p.is_relative_to(root):
            raise ValueError('Path must remain inside the project')
        return p

    def export(self, project, progress, queue=None):
        q = queue or self.queue(project)
        if not isinstance(progress, dict) or set(progress.get('tasks', {})) != {t['id'] for t in q['tasks']}:
            raise ValueError('Progress tasks differ from the active queue')
        records, completed = [], []
        for t in q['tasks']:
            d = progress['tasks'][t['id']]
            if type(d.get('done')) is not bool or not isinstance(d.get('segments'), list) or not d['segments']:
                raise ValueError('Invalid draft')
            previous = 0
            for i, s in enumerate(d['segments']):
                a, b = s.get('start_frame'), s.get('end_frame_exclusive')
                if type(a) is not int or type(b) is not int or a != previous or not 0 <= a < b <= t['clip_frames']:
                    raise ValueError('Draft intervals must cover the clip continuously')
                previous = b
                phase = s.get('phase')
                if phase not in [*PHASES, '无法判断', '混合/需调整区间', '无效动作', '']:
                    raise ValueError('Unknown phase')
                if not d['done']:
                    continue
                if not phase:
                    raise ValueError('Completed clips must have explicit phases')
                kind = 'single' if phase in PHASES else {'无效动作':'invalid', '混合/需调整区间':'mixed', '无法判断':'uncertain'}[phase]
                stroke = s.get('stroke_type') or '无法判断'
                records.append(dict(id=t['id']+'--'+str(i+1).zfill(2), parent_candidate_id=t['id'],
                    clip_id=t['clip_id'], split=t['split'], source_sha256=t['source_sha256'],
                    normalized_video_sha256=t['normalized_video_sha256'], fps=q['fps'], clip_frames=t['clip_frames'],
                    video=t['video'], start_frame=a, end_frame_exclusive=b, start_seconds=a/q['fps'], end_seconds=b/q['fps'],
                    phase=phase if kind=='single' else None, annotation_label=phase, annotation_type=kind,
                    phase_usable=kind=='single', excluded=kind=='invalid', reviewed=True, review_status='human_reviewed',
                    shot_type=s.get('shot_type') or '无法判断', shot_type_usable=s.get('shot_type') not in [None,'','无法判断','混合/需调整区间','无效动作'] and kind!='invalid',
                    stroke_type=stroke, stroke_type_reviewed=bool(s.get('stroke_type_reviewed')),
                    stroke_type_usable=bool(s.get('stroke_type_reviewed')) and phase in ['引拍','挥拍','随挥'] and stroke in ['正手','单手反手','双手反手'],
                    stroke_type_source=s.get('stroke_type_source'), notes=s.get('notes',''),
                    prior_record_id=s.get('prior_record_id'), continuity_sources=s.get('continuity_sources'),
                    phase_schema_version=q['phase_schema']['version'], annotation_batch_id=q['queue_id']))
            if previous != t['clip_frames']:
                raise ValueError('Draft does not cover the whole clip')
            if d['done']:
                completed.append(t['id'])
        payload = {k: deepcopy(q.get(k)) for k in ['manifest_sha256','baseline_checkpoint_sha256','fps',
            'candidate_registry','candidate_registry_sha256','baseline_labels_sha256','proposal_model_sha256',
            'excluded_clips','clip_type_confirmations','curation_sha256','phase_schema','queue_id']}
        payload.update(schema_version=3, annotation_mode='complete_clip_intervals', created_at=now(),
            scope=q['export_scope'], completed_tasks=completed, annotation_progress=deepcopy(progress), records=records)
        root = self.project(project)
        validate_export(root, annotation_registry(root,payload), payload)
        return payload

    def annotations(self, project):
        q = self.queue(project)
        with self.connect() as db:
            row = db.execute('SELECT * FROM annotations WHERE project=?', (project,)).fetchone()
        if row:
            if row['queue_id'] != q['queue_id']:
                raise Conflict('The active model changed; reload the workbench')
            return {**dict(row), 'payload': json.loads(row['payload'])}
        payload = self.export(project, q['initial_progress'], q)
        return self.save(project, q['queue_id'], 0, payload['annotation_progress'])

    def save(self, project, queue_id, revision, progress):
        q = self.queue(project)
        if queue_id != q['queue_id']:
            raise Conflict('This editor belongs to an older model; reload before saving')
        payload = self.export(project, progress, q)
        digest = progress_hash(progress)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM annotations WHERE project=?',(project,)).fetchone()
            if row and row['progress_hash'] == digest and row['queue_id'] == queue_id:
                return {**dict(row), 'payload': json.loads(row['payload'])}
            if (row['revision'] if row else 0) != revision:
                raise Conflict('Another page saved newer annotations. Export your draft, then reload.')
            revision += 1
            saved = now()
            # Immutable file first; SQLite commits the revision only after it is recoverable.
            atomic_json(self.directory/'annotations'/project/f'{revision:06d}-{digest}.json',payload)
            db.execute('INSERT OR REPLACE INTO annotations VALUES (?,?,?,?,?,?)',
                (project,queue_id,revision,json.dumps(payload,ensure_ascii=False),saved,digest))
        return {'project':project,'queue_id':queue_id,'revision':revision,'saved_at':saved,'progress_hash':digest,'payload':payload}

    def models(self, project):
        root = self.project(project)
        result = []
        for p in root.glob('semantic-pose-*/summary.json'):
            s = json.loads(p.read_text())
            if s.get('status') == 'complete' and s.get('training_type') == 'fully_supervised':
                result.append({'version':p.parent.name,'training_frames':s['training_frames'],
                    'human_clips':s['human_complete_clips'],'model_sha256':s['model_sha256'],
                    'accuracy':s['evaluation']['oof_metrics']['accuracy'],
                    'macro_f1':s['evaluation']['oof_metrics']['macro_f1'],
                    'created_at':s.get('created_at'),'report_url':f'/projects/{project}/{p.parent.name}/'})
        return result

    def model(self, project, version=None):
        version = version or self.current(project)['version']
        if not re.fullmatch(r'semantic-pose-[A-Za-z0-9_-]+', version):
            raise ValueError('Invalid model version')
        p = self.inside(self.project(project),version)
        s = json.loads((p/'summary.json').read_text())
        if s.get('status') != 'complete' or s.get('training_type') != 'fully_supervised':
            raise ValueError('A completed supervised model is required')
        return p,s

    def new_job(self, project, action, request):
        self.project(project)
        if action not in ['train','import_video','remote_pose']:
            raise ValueError('Unknown action')
        identifier = uuid.uuid4().hex
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute("SELECT id FROM jobs WHERE project=? AND status IN ('queued','running')",(project,)).fetchone():
                raise Conflict('This project already has a running task')
            stamp=now()
            db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,NULL,?,NULL,NULL)',
                (identifier,project,action,'queued',stamp,stamp,json.dumps(request,ensure_ascii=False)))
        return self.job(identifier)

    def update_job(self, identifier, **changes):
        allowed={'status','pid','result','error'}
        if not changes or set(changes)-allowed:raise ValueError('Invalid job update')
        changes['updated_at']=now()
        if 'result' in changes:changes['result']=json.dumps(changes['result'],ensure_ascii=False)
        with self.connect() as db:
            db.execute('UPDATE jobs SET '+','.join(k+'=?' for k in changes)+' WHERE id=?',(*changes.values(),identifier))

    def job(self, identifier):
        if not re.fullmatch(r'[a-f0-9]{32}',identifier):raise ValueError('Invalid job ID')
        with self.connect() as db:
            row=db.execute('SELECT * FROM jobs WHERE id=?',(identifier,)).fetchone()
        if not row:raise ValueError('Task not found')
        value=dict(row)
        for k in ['request','result']:
            value[k]=json.loads(value[k]) if value[k] else None
        return value

    def jobs(self, project):
        with self.connect() as db:
            return [self.job(r['id']) for r in db.execute('SELECT id FROM jobs WHERE project=? ORDER BY created_at DESC LIMIT 30',(project,))]
