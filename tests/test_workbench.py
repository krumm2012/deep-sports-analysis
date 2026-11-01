"""Durability, split protection and API boundaries for the local workflow."""
from copy import deepcopy
from functools import partial
import io
import json
from pathlib import Path
from threading import Thread
import tarfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from dsa.measure.semantic_pilot import sha
from dsa.workbench.app import Workbench, make_server
from dsa.workbench.backend import safe_extract
from dsa.workbench.store import Store, Conflict, atomic_json


@pytest.fixture
def project(tmp_path):
    root=tmp_path/'projects/demo';editor=root/'semantic-pose-v5/edit';editor.mkdir(parents=True)
    atomic_json(root/'manifest.json',{'fps':25,'clips':[{'id':'a','split':'train','frames':100,'fps':25}, {'id':'v','split':'validation','frames':100,'fps':25}]})
    task={'id':'clip-a','clip_id':'a','split':'train','fps':25,'clip_frames':100,'source_sha256':'raw-sha','normalized_video_sha256':'video-sha','video':'../../clips/a/source.mp4'}
    registry={'fps':25,'items':[task]};atomic_json(editor/'candidates.json',registry)
    progress={'active':0,'tasks':{'clip-a':{'done':True,'segments':[{'start_frame':0,'end_frame_exclusive':100,'phase':'准备','shot_type':'无法判断'}]}}}
    queue={'queue_id':'test-queue','tasks':[task],'initial_progress':progress,'fps':25,
           'manifest_sha256':sha(root/'manifest.json'),'candidate_registry':'semantic-pose-v5/edit/candidates.json',
           'candidate_registry_sha256':sha(editor/'candidates.json'),'phase_schema':{'version':'tennis-five-phase-v1'},'export_scope':'training-only'}
    atomic_json(editor/'queue.json',queue)
    atomic_json(root/'current-semantic.json',{'version':'semantic-pose-v5','editor':'semantic-pose-v5/edit/','report':'semantic-pose-v5/'})
    return tmp_path


def test_annotations_are_durable_and_conflicting_browser_cannot_overwrite(project):
    store=Store(project);first=store.annotations('demo');assert first['revision']==1
    draft=deepcopy(first['payload']['annotation_progress']);draft['tasks']['clip-a']['segments'][0]['phase']='引拍'
    saved=store.save('demo','test-queue',1,draft);assert saved['revision']==2
    restarted=Store(project);assert restarted.annotations('demo')['payload']['annotation_progress']==draft
    stale=deepcopy(draft);stale['tasks']['clip-a']['segments'][0]['phase']='挥拍'
    with pytest.raises(Conflict):restarted.save('demo','test-queue',1,stale)
    assert restarted.annotations('demo')['payload']['records'][0]['phase']=='引拍'
    assert len(list((store.directory/'annotations/demo').glob('*.json')))==2


def test_draft_not_confirmed_does_not_enter_training_records(project):
    s=Store(project);a=s.annotations('demo');p=deepcopy(a['payload']['annotation_progress']);p['tasks']['clip-a']['done']=False
    a=s.save('demo','test-queue',1,p);assert a['payload']['records']==[]
    app=Workbench(project)
    with pytest.raises(ValueError,match='Confirm every'):app.train('demo')


def test_unknown_tasks_frame_gaps_and_stale_queue_rejected(project):
    s=Store(project);a=s.annotations('demo');p=deepcopy(a['payload']['annotation_progress'])
    p['tasks']['clip-v']=deepcopy(p['tasks']['clip-a'])
    with pytest.raises(ValueError,match='tasks differ'):s.save('demo','test-queue',1,p)
    p=deepcopy(a['payload']['annotation_progress']);p['tasks']['clip-a']['segments'][0]['start_frame']=1
    with pytest.raises(ValueError,match='continuously'):s.save('demo','test-queue',1,p)
    with pytest.raises(Conflict):s.save('demo','old-model-queue',1,a['payload']['annotation_progress'])


def test_training_snapshot_is_immutable_and_duplicate_job_rejected(project,monkeypatch):
    app=Workbench(project);a=app.store.annotations('demo');monkeypatch.setattr(app,'launch',lambda job:job)
    job=app.train('demo');snapshot=app.store.directory/'jobs'/job['id']/'annotation.json';before=snapshot.read_bytes()
    p=deepcopy(a['payload']['annotation_progress']);p['tasks']['clip-a']['segments'][0]['phase']='随挥'
    app.store.save('demo','test-queue',1,p)
    assert snapshot.read_bytes()==before
    assert json.loads(before)['records'][0]['phase']=='准备'
    with pytest.raises(Conflict):app.train('demo')
    app.store.update_job(job['id'],status='failed',error='simulated')
    retry=app.retry(job['id']);assert retry['id']!=job['id']
    assert (app.store.directory/'jobs'/retry['id']/'annotation.json').read_bytes()==before


def test_stopped_worker_is_retryable_after_server_restart(project):
    s=Store(project);job=s.new_job('demo','train',{});s.update_job(job['id'],pid=2**30,status='running')
    restarted=Workbench(project);assert restarted.store.job(job['id'])['status']=='interrupted'


@pytest.fixture
def http(project):
    server=make_server(project,0);thread=Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}',server.workbench
    finally:server.shutdown();server.server_close();thread.join()


def test_api_persists_export_and_rejects_cross_origin_and_private_paths(http):
    base,app=http
    with urlopen(base+'/api/health') as r:assert json.load(r)['service']=='dsa-workbench'
    with urlopen(base+'/api/projects/demo/annotations') as r:a=json.load(r)
    progress=deepcopy(a['payload']['annotation_progress']);progress['tasks']['clip-a']['segments'][0]['notes']='人工修改'
    body=json.dumps({'queue_id':'test-queue','revision':1,'progress':progress}).encode()
    path=base+'/api/projects/demo/annotations'
    with pytest.raises(HTTPError) as err:urlopen(Request(path,data=body,method='PUT'))
    assert err.value.code==403
    with pytest.raises(HTTPError) as err:urlopen(Request(path,data=body,method='PUT',headers={'X-Workbench-Request':'1','Origin':'https://other.example'}))
    assert err.value.code==403
    with urlopen(Request(path,data=body,method='PUT',headers={'X-Workbench-Request':'1'})) as r:assert json.load(r)['revision']==2
    with urlopen(base+'/api/projects/demo/export') as r:assert json.load(r)['records'][0]['notes']=='人工修改'
    with pytest.raises(HTTPError) as err:urlopen(base+'/projects/../.workbench/state.sqlite3')
    assert err.value.code in [403,404]


def test_backend_archive_traversal_is_rejected_before_any_write(tmp_path):
    f=io.BytesIO()
    with tarfile.open(fileobj=f,mode='w:gz') as tar:
        info=tarfile.TarInfo('../escape');info.size=1;tar.addfile(info,io.BytesIO(b'x'))
    f.seek(0)
    with pytest.raises(ValueError,match='Unsafe'):safe_extract(f,tmp_path)
    assert not list(tmp_path.iterdir())


def test_report_editor_links_return_to_durable_source(http):
    from http.client import HTTPConnection
    from urllib.parse import urlsplit
    base,_=http
    c=HTTPConnection(urlsplit(base).netloc)
    c.request('GET','/projects/demo/semantic-pose-v5/edit/?clip=a')
    response=c.getresponse()
    assert response.status==302
    assert response.getheader('Location')=='/workbench/editor?project=demo&clip=a'
    response.read();c.close()


def test_launchagent_restart_retries_registration_transition(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from dsa.workbench import mac_service
    calls=[]
    def command(args,**kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=5 if len(calls)==1 else 0,stderr='busy')
    monkeypatch.setattr(mac_service.subprocess,'run',command)
    monkeypatch.setattr(mac_service.time,'sleep',lambda seconds:None)
    mac_service.bootstrap('gui/501',tmp_path/'service.plist')
    assert len(calls)==2


def test_early_worker_exit_does_not_leave_task_running(project,monkeypatch):
    import time
    from dsa.workbench import app as module
    app=Workbench(project);job=app.store.new_job('demo','train',{})
    class EarlyExit:
        pid=2**30
        def wait(self):return 7
    monkeypatch.setattr(module.subprocess,'Popen',lambda *args,**kwargs:EarlyExit())
    app.launch(job)
    for _ in range(100):
        saved=app.store.job(job['id'])
        if saved['status']=='failed':break
        time.sleep(.005)
    assert saved['status']=='failed' and 'exit 7' in saved['error']
