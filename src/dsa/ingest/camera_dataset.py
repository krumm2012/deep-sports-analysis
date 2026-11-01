"""Prepare independent camera-drill clips and extract body-only MoSeq inputs.

The ZIP stays untouched. Detection and pose use separate GPU lifetimes; no ball,
no court calibration, and no broadcast stroke classifier are run here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import hashlib
import json
import importlib.metadata
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import time
from zipfile import ZipFile

import cv2
import numpy as np

from dsa.measure.drill import export
from dsa.scene import tracks

ROI = (.35, .08, .98, .97)
FPS = 25


def save_provenance(root: Path, phase: str, arguments: list[str]):
    from dsa.paths import MODELS

    root.mkdir(parents=True, exist_ok=True)
    weights = [MODELS / 'rfdetr/rf-detr-medium.pth', MODELS / 'rtmw/rtmw.onnx']
    packages = {}
    for name in ['torch', 'rfdetr', 'rtmlib', 'onnxruntime-gpu', 'numpy', 'scipy', 'opencv-python', 'supervision']:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    record = {'phase': phase, 'arguments': arguments, 'software': packages,
              'weights': [{'path': str(p), 'sha256': sha256(p), 'bytes': p.stat().st_size} for p in weights if p.exists()],
              'models_retrained': [], 'detector': 'RF-DETR Medium, resolution 1152, batch4',
              'pose': 'RTMW-x, ONNX CUDA, 384x288 input, batch<=32; 19 body/foot points used for MoSeq',
              'gpu_lifetimes': 'all detection clips first; release PyTorch detector/cache; then one shared RTMW session'}
    write_json(root / f'{phase}-provenance.json', record)


def write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + '.part')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def clip_seconds(name: str) -> int:
    parts = Path(name).stem.split('.')
    if len(parts) != 2 or not all(x.isdigit() for x in parts) or not 0 <= int(parts[1]) < 60:
        raise ValueError(f'Expected minute.second.mp4 name: {name}')
    return int(parts[0]) * 60 + int(parts[1])


def safe_video_members(z: ZipFile):
    out = []
    for info in z.infolist():
        p = PurePosixPath(info.filename)
        # Reject unsafe entries even if they are not media, rather than silently
        # allowing an archive that could escape the dataset destination.
        if p.is_absolute() or '..' in p.parts or '\\' in info.filename or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError(f'Unsafe ZIP entry: {info.filename}')
        if not info.is_dir() and p.suffix.lower() == '.mp4' and '__MACOSX' not in p.parts and not p.name.startswith('._'):
            out.append(info)
    names = [PurePosixPath(i.filename).name for i in out]
    if len(names) != len(set(names)):
        raise ValueError('Archive contains duplicate video basenames')
    return sorted(out, key=lambda i: clip_seconds(PurePosixPath(i.filename).name))


def read_manifest(root):
    return json.loads((root / 'manifest.json').read_text())


def event(root: Path, stage: str, clip: str, **fields):
    value = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'stage': stage, 'id': clip, **fields}
    with (root / 'progress.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')
    print(json.dumps(value, ensure_ascii=False), flush=True)


def update_clip(root: Path, clip_id: str, **fields):
    manifest = read_manifest(root)
    for clip in manifest['clips']:
        if clip['id'] == clip_id:
            clip.update(fields)
            break
    write_json(root / 'manifest.json', manifest)


def probe(path: Path):
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_packets',
           '-show_entries', 'stream=width,height,r_frame_rate,avg_frame_rate,duration,nb_frames:packet=pts_time',
           '-of', 'json', str(path)]
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    d = json.loads(r.stdout); stream = d['streams'][0]
    pts = np.sort(np.array([float(f['pts_time']) for f in d['packets'] if 'pts_time' in f]))
    if stream.get('nb_frames') and int(stream['nb_frames']) != len(pts):
        raise ValueError('Packet/frame counts disagree; review original timestamps before preparation')
    if len(pts) < 20 or np.any(np.diff(pts) < 0):
        raise ValueError('Unreadable or nonmonotonic video timestamps')
    pts -= pts[0]
    duration = float(stream.get('duration', pts[-1] + np.median(np.diff(pts))))
    return stream, pts.tolist(), duration


def make_contact_sheet(root: Path, clips: list):
    picks = np.unique(np.linspace(0, len(clips) - 1, min(12, len(clips))).round().astype(int))
    tiles = []
    for i in picks:
        c = clips[i]; cap = cv2.VideoCapture(str(root / c['raw']))
        cap.set(cv2.CAP_PROP_POS_MSEC, c['duration_seconds'] * 500)
        ok, frame = cap.read(); cap.release()
        if not ok:
            frame = np.zeros((720, 1280, 3), np.uint8)
        tile = cv2.resize(frame, (480, 270))
        cv2.putText(tile, c['filename'], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    image = np.vstack([np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)])
    cv2.imwrite(str(root / 'scene-contact-sheet.jpg'), image)


def make_pose_contact_sheet(root: Path):
    from dsa.pages.moseq_report import BONES

    clips = read_manifest(root)['clips']
    picks = np.unique(np.linspace(0, len(clips) - 1, min(12, len(clips))).round().astype(int))
    tiles = []
    for i in picks:
        c = clips[i]; folder = root / 'clips' / c['id']
        cap = cv2.VideoCapture(str(root / c['video'])); frame_i = c['frames'] // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i); ok, frame = cap.read(); cap.release()
        if not ok: frame = np.zeros((720, 1280, 3), np.uint8)
        path = folder / 'moseq-input.npz'
        if c.get('pose_status') == 'complete' and path.exists():
            with np.load(path, allow_pickle=False) as z:
                xy, confidence = z['image_coordinates'][frame_i], z['confidences'][frame_i]
            good = np.isfinite(xy).all(1) & (confidence >= .1)
            for a, b in BONES:
                if good[a] and good[b]:
                    cv2.line(frame, tuple(np.rint(xy[a]).astype(int)), tuple(np.rint(xy[b]).astype(int)), (0, 255, 255), 4, cv2.LINE_AA)
            status = '25fps | accepted'
        else:
            status = 'excluded: no reliable main-player input'
        tile = cv2.resize(frame, (480, 270))
        cv2.putText(tile, c['filename'], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 2)
        cv2.putText(tile, status, (10, 255), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 255), 1)
        tiles.append(tile)
    while len(tiles) % 3: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(root / 'pose-quality-contact-sheet.jpg'), np.vstack([np.hstack(tiles[j:j+3]) for j in range(0, len(tiles), 3)]))


def standardize(root: Path, clip: dict):
    start = time.perf_counter(); d = root / 'clips' / clip['id']; d.mkdir(parents=True, exist_ok=True)
    source = root / clip['raw']; target = d / 'source.mp4'
    if not target.exists():
        tmp = d / 'source.part.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-threads', '2', '-i', str(source),
                        '-vf', f'fps={FPS},scale=1280:720', '-an', '-c:v', 'libx264', '-threads', '2',
                        '-crf', '18', '-preset', 'veryfast', '-movflags', '+faststart', str(tmp)], check=True)
        tmp.replace(target)
    cap = cv2.VideoCapture(str(target)); frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); actual_fps = float(cap.get(cv2.CAP_PROP_FPS)); cap.release()
    if abs(actual_fps - FPS) > .001:
        raise ValueError(f'Expected {FPS}fps source; got {actual_fps}')
    original_pts = np.asarray(clip['original_pts_seconds'])
    nearest_original = np.abs(np.arange(frames)[:, None] / FPS - original_pts[None]).argmin(axis=1)
    represented = int(len(np.unique(nearest_original)))
    if frames < 20:
        raise ValueError('Standardized source is too short')
    provenance = {'source_file': str(source.resolve()), 'source_sha256': clip['sha256'],
                  'archive': str(root / 'archive-reference.json'), 'archive_member': clip['archive_member'],
                  'original_frames': clip['original_frames'], 'original_pts_seconds': clip['original_pts_seconds'],
                  'original_resolution': clip['original_resolution'], 'duration_seconds': clip['duration_seconds'],
                  'timestamp_source': 'sorted original presentation timestamps (PTS), not nominal 50fps',
                  'nearest_pts_mapping_unique_original_frames': represented,
                  'nearest_pts_mapping_duplicated_output_frames': frames - represented,
                  'nearest_pts_mapping_unrepresented_original_frames': len(original_pts) - represented,
                  'processed_fps': FPS, 'processed_resolution': [1280, 720], 'processed_frames': frames,
                  'frame_mapping': 'nearest original PTS; resampling is not extra independent data',
                  'scene': 'fixed high oblique camera, enclosed tennis drill lane; no full court homography',
                  'person_identity': 'anonymous main player', 'scope': 'independent clip in a multi-video dataset'}
    write_json(d / 'source-provenance.json', provenance)
    return {'frames': frames, 'fps': FPS, 'preparation_status': 'complete', 'preparation_seconds': time.perf_counter() - start}


def prepare(archive: Path, root: Path, workers: int = 4):
    root.mkdir(parents=True, exist_ok=True); raw = root / 'raw'; raw.mkdir(exist_ok=True)
    if (root / 'manifest.json').exists():
        manifest = read_manifest(root)
        if manifest.get('fps') != FPS:
            raise ValueError('Existing dataset FPS differs; archive derived files before rebuilding')
        if manifest['archive_sha256'] != sha256(archive):
            raise ValueError('Dataset directory already belongs to a different archive')
    else:
        archive_hash = sha256(archive)
        write_json(root / 'archive-reference.json', {'path': str(archive.resolve()), 'bytes': archive.stat().st_size, 'sha256': archive_hash})
        clips = []; seen = {}
        with ZipFile(archive) as z:
            members = safe_video_members(z)
            for info in members:
                name = PurePosixPath(info.filename).name; path = raw / name
                if not path.exists():
                    tmp = path.with_suffix('.mp4.part')
                    with z.open(info) as src, tmp.open('wb') as dst:
                        while b := src.read(1 << 20):
                            dst.write(b)
                    tmp.replace(path)
                h = sha256(path); stream, pts, duration = probe(path)
                c = {'id': Path(name).stem.replace('.', '_'), 'filename': name, 'archive_member': info.filename,
                     'raw': str(path.relative_to(root)), 'sha256': h, 'bytes': path.stat().st_size,
                     'start_seconds': clip_seconds(name), 'duration_seconds': duration,
                     'original_frames': len(pts), 'original_pts_seconds': pts,
                     'original_resolution': [stream['width'], stream['height']],
                     'input': f"clips/{Path(name).stem.replace('.', '_')}/moseq-input.npz",
                     'video': f"clips/{Path(name).stem.replace('.', '_')}/source.mp4", 'split': 'pending', 'pose_status': 'pending'}
                if h in seen:
                    c.update(split='excluded', exclusion_reason=f'duplicate SHA256 of {seen[h]}')
                else:
                    seen[h] = c['id']
                clips.append(c)
        manifest = {'archive': str(archive.resolve()), 'archive_sha256': archive_hash,
                    'schema': 'camera-drill-dataset-v1', 'roi_fraction': list(ROI),
                    'fps': FPS, 'normalization': 'same as dsa.measure.drill; constant clip median hip/torso',
                    'clips': clips, 'video_count': len(clips),
                    'total_original_seconds': sum(c['duration_seconds'] for c in clips),
                    'total_original_frames': sum(c['original_frames'] for c in clips)}
        write_json(root / 'manifest.json', manifest)
        make_contact_sheet(root, clips)
        event(root, 'inventory', '*', clips=len(clips), seconds=manifest['total_original_seconds'])
    eligible = [c for c in manifest['clips'] if c['split'] != 'excluded' and c.get('preparation_status') != 'complete']
    with ThreadPoolExecutor(workers) as pool:
        jobs = {pool.submit(standardize, root, c): c for c in eligible}
        for job in as_completed(jobs):
            c = jobs[job]
            try:
                fields = job.result(); update_clip(root, c['id'], **fields); event(root, 'prepare', c['id'], **fields)
            except Exception as exc:
                update_clip(root, c['id'], split='excluded', preparation_status='failed', exclusion_reason=f'{type(exc).__name__}: {exc}')
                event(root, 'prepare', c['id'], status='failed', error=str(exc))
    return read_manifest(root)


def choose_subject(boxes, ids, fps, width, height, roi=ROI):
    people = tracks.people(boxes, ids, fps)
    x0, y0, x1, y1 = np.array(roi) * [width, height, width, height]
    ranking = []
    for person, rec in people['people'].items():
        valid = []
        for frame in rec['frames']:
            js = [j for j, t in enumerate(ids[frame]) if t >= 0 and people['person_of'][int(t)] == person]
            if not js: continue
            b = boxes[frame][max(js, key=lambda j: boxes[frame][j, 4])]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1 and b[3] - b[1] >= .2 * height:
                valid.append(float(b[3] - b[1]))
        ranking.append({'person': int(person), 'roi_frames': len(valid), 'score': float(len(valid) * np.median(valid)) if valid else 0})
    if not ranking or max(c['score'] for c in ranking) <= 0:
        raise ValueError('No persistent main player inside ROI')
    best = max(ranking, key=lambda c: c['score'])
    if best['roi_frames'] / len(boxes) < .8:
        raise ValueError(f"Main-player coverage {best['roi_frames']}/{len(boxes)} is below 80%")
    wanted = {}
    for i, (b, ts) in enumerate(zip(boxes, ids)):
        js = [j for j, t in enumerate(ts) if t >= 0 and people['person_of'][int(t)] == best['person']]
        if js: wanted[i] = max(js, key=lambda j: b[j, 4])
    return wanted, best, ranking


def detect(root: Path, batch: int = 4):
    import torch
    from dsa.paths import MODELS
    os.environ.setdefault('RF_HOME', str(MODELS / 'rfdetr'))
    from dsa.gpu.models import Detector
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; CPU fallback forbidden')
    pending = [c for c in read_manifest(root)['clips'] if c.get('preparation_status') == 'complete' and c.get('detection_status') != 'complete']
    if not pending: return
    det = Detector('cuda')
    for c in pending:
        t = time.perf_counter(); folder = root / 'clips' / c['id']; cap = cv2.VideoCapture(str(folder / 'source.mp4'))
        try:
            boxes, fs = [], []
            while True:
                ok, frame = cap.read()
                if not ok: break
                fs.append(frame)
                if len(fs) == batch:
                    boxes.extend(det.people(fs)); fs.clear()
            if fs: boxes.extend(det.people(fs))
            if len(boxes) != c['frames']: raise ValueError('Detection frame count differs from source')
            ids = tracks.track(boxes, FPS)
            np.savez_compressed(folder / 'detections.npz', n=np.array([len(b) for b in boxes]), boxes=np.concatenate(boxes), track=np.concatenate(ids),
                                source_sha256=c['sha256'])
            wanted, chosen, ranked = choose_subject(boxes, ids, FPS, 1280, 720)
            fields = {'detection_status': 'complete', 'main_player_coverage': len(wanted) / len(boxes),
                      'chosen_subject': chosen, 'candidate_tracks': ranked, 'detection_seconds': time.perf_counter() - t}
            update_clip(root, c['id'], **fields); event(root, 'detect', c['id'], **fields)
        except Exception as exc:
            update_clip(root, c['id'], split='excluded', detection_status='failed', exclusion_reason=f'{type(exc).__name__}: {exc}')
            event(root, 'detect', c['id'], status='failed', error=str(exc))
        finally: cap.release()
    del det; gc.collect(); torch.cuda.empty_cache()


def pose(root: Path, pose_batch: int = 32):
    import torch
    from dsa.gpu.models import Pose, KP
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; CPU fallback forbidden')
    pending = [c for c in read_manifest(root)['clips'] if c.get('detection_status') == 'complete' and c.get('pose_status') != 'complete']
    if not pending: return
    model = Pose('cuda')
    with ThreadPoolExecutor(8) as pool:
        for c in pending:
            t = time.perf_counter(); folder = root / 'clips' / c['id']; cap = cv2.VideoCapture(str(folder / 'source.mp4'))
            try:
                with np.load(folder / 'detections.npz', allow_pickle=False) as z:
                    ns=z['n']; flat_boxes=z['boxes']; flat_ids=z['track']; offsets=np.cumsum(ns)[:-1]
                    boxes=np.split(flat_boxes, offsets); ids=np.split(flat_ids, offsets)
                wanted, chosen, ranked = choose_subject(boxes, ids, FPS, 1280, 720)
                kp = [np.full((len(b), len(KP), 3), np.nan, np.float16) for b in boxes]
                batch, indices = [], []
                def flush():
                    if batch:
                        assert len(batch) <= pose_batch
                        out = model.run(batch)
                        for (i,j), result in zip(indices,out): kp[i][j]=result
                        batch.clear(); indices.clear()
                frame_i=0
                while True:
                    ok, frame=cap.read()
                    if not ok: break
                    if frame_i in wanted:
                        j=wanted[frame_i];batch.append(model.prepare(frame,boxes[frame_i][j]));indices.append((frame_i,j))
                        if len(batch) == pose_batch: flush()
                    frame_i+=1
                flush()
                if frame_i != len(boxes): raise ValueError('Pose frame count differs from detections')
                np.savez_compressed(folder/'raw.npz',n=ns,boxes=flat_boxes,track=flat_ids,kp=np.concatenate(kp),fps=FPS,width=1280,height=720,
                                    seconds=json.dumps({'pose':time.perf_counter()-t}),models_run='RF-DETR person detector; RTMW body pose; no ball/court/stroke model')
                out=export(folder, roi=ROI)
                quality=json.loads((folder/'pose-quality.json').read_text())
                fields={'pose_status':'complete','input_sha256':sha256(out),'pose_seconds':time.perf_counter()-t,
                        'raw_observed_keypoint_fraction':quality['raw_observed_keypoint_fraction'],
                        'zero_confidence_fraction':quality['zero_confidence_fraction']}
                update_clip(root,c['id'],**fields);event(root,'pose',c['id'],**fields)
            except Exception as exc:
                update_clip(root,c['id'],split='excluded',pose_status='failed',exclusion_reason=f'{type(exc).__name__}: {exc}')
                event(root,'pose',c['id'],status='failed',error=str(exc))
            finally:cap.release()
    del model;gc.collect();torch.cuda.empty_cache()
    make_pose_contact_sheet(root)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','detect','pose','all']);p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',type=Path);p.add_argument('--workers',type=int,default=4);p.add_argument('--batch',type=int,default=4);p.add_argument('--pose-batch',type=int,default=32)
    a=p.parse_args()
    if min(a.workers,a.batch,a.pose_batch)<1:p.error('Batch/workers values must be positive')
    import sys
    save_provenance(a.root, a.phase, sys.argv[1:])
    if a.phase in ('prepare','all'):
        if not a.archive:p.error('--archive is required for preparation')
        prepare(a.archive,a.root,a.workers)
    if a.phase in ('detect','all'):detect(a.root,a.batch)
    if a.phase in ('pose','all'):pose(a.root,a.pose_batch)


if __name__=='__main__':main()
