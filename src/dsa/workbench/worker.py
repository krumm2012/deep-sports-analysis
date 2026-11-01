"""Detached workers keep task state and logs across browser/server restarts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

from dsa.workbench.store import Store, atomic_json


def train(store, job):
    from dsa.measure.semantic_supervised import run
    from dsa.pages.semantic_pose_report import report
    from dsa.pages.semantic_pose_edit import build
    root=store.project(job['project'])
    output=root/('semantic-pose-wb-'+job['id'][:12])
    annotation=store.directory/'jobs'/job['id']/'annotation.json'
    summary=run(root,output,annotation)
    report(root,output)
    build(root,output,output/'edit',annotation)
    return {'version':output.name,'training_frames':summary['training_frames'],
            'human_clips':summary['human_complete_clips'],
            'report_url':f"/projects/{job['project']}/{output.name}/",'activated':False}


def import_video(store, job):
    from dsa.ingest.camera_dataset import probe, standardize, sha256
    root=store.directory/'imports'/job['id']
    source=root/'raw/original.mp4'
    if not source.exists() and job['request'].get('source_import_id'):
        import shutil
        old=store.job(job['request']['source_import_id'])
        if old['project']!=job['project']:raise ValueError('Import belongs to another project')
        source.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(store.directory/'imports'/old['id']/'raw/original.mp4',source)
    stream,pts,duration=probe(source)
    if duration>120:raise ValueError('First release accepts fixed-camera clips up to 120 seconds')
    clip={'id':'video','filename':job['request']['filename'],'raw':'raw/original.mp4',
          'archive_member':job['request']['filename'],'sha256':sha256(source),
          'bytes':source.stat().st_size,'start_seconds':0,'duration_seconds':duration,
          'original_frames':len(pts),'original_pts_seconds':pts,
          'original_resolution':[stream['width'],stream['height']],
          'video':'clips/video/source.mp4','input':'clips/video/moseq-input.npz',
          'split':'inference','pose_status':'pending'}
    atomic_json(root/'archive-reference.json',{'source':'workbench_upload','sha256':clip['sha256']})
    clip.update(standardize(root,clip))
    atomic_json(root/'manifest.json',{'fps':25,'clips':[clip],'purpose':'inference_only; not part of training or validation'})
    return {'import_id':job['id'],'filename':clip['filename'],'frames':clip['frames'],'fps':25,
            'video_url':f"/api/imports/{job['id']}/video",'pose_ready':False,'split':'inference'}


def remote_pose(store, job):
    import io
    import shlex
    import tarfile
    import tempfile
    from dsa.workbench.backend import read_backend, ssh_command, safe_extract
    settings=read_backend(Path(job['request']['backend_config']))
    identifier=job['request']['import_id']
    item=store.job(identifier)
    if item['project']!=job['project'] or item['action']!='import_video' or item['status']!='succeeded':
        raise ValueError('A prepared import from this project is required')
    root=store.directory/'imports'/identifier
    remote_root=settings['project_directory']
    relative='data/workbench-incoming/'+identifier
    local_code=Path(__file__).with_name('remote_pose.py')
    with tempfile.TemporaryFile() as archive:
        with tarfile.open(fileobj=archive,mode='w:gz') as tar:
            for name in ['manifest.json','clips/video/source.mp4','clips/video/source-provenance.json']:
                tar.add(root/name,arcname=relative+'/'+name)
            tar.add(local_code,arcname='src/dsa/workbench_remote_pose.py')
        archive.seek(0)
        subprocess.run(ssh_command(settings,'cd '+shlex.quote(remote_root)+' && tar -xzf -'),stdin=archive,check=True,timeout=300)
    command='cd '+shlex.quote(remote_root)+' && PYTHONPATH=src '+shlex.quote(settings['python'])+' src/dsa/workbench_remote_pose.py '+shlex.quote(relative)
    subprocess.run(ssh_command(settings,command),check=True,timeout=1800)
    command='cd '+shlex.quote(remote_root+'/'+relative)+' && tar -czf - manifest.json clips/video/moseq-input.npz clips/video/pose-quality.json'
    with tempfile.TemporaryFile() as received:
        subprocess.run(ssh_command(settings,command),stdout=received,check=True,timeout=300)
        received.seek(0);safe_extract(received,root)
    manifest=json.loads((root/'manifest.json').read_text());clip=manifest['clips'][0]
    if clip.get('pose_status')!='complete':raise ValueError(clip.get('exclusion_reason','Pose extraction failed'))
    from dsa.measure.moseq_local import read_input
    data=read_input(root/clip['input'])
    if len(data['coordinates'])!=clip['frames'] or data['fps']!=25:raise ValueError('Returned pose alignment differs')
    return {'import_id':identifier,'pose_ready':True,'frames':clip['frames'],'backend':'5090',
            'video_url':f'/api/imports/{identifier}/video','split':'inference'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--job',required=True);a=p.parse_args()
    store=Store(a.data);job=store.job(a.job)
    try:
        store.update_job(a.job,status='running')
        print('Starting',job['action'],flush=True)
        result={'train':train,'import_video':import_video,'remote_pose':remote_pose}[job['action']](store,job)
        store.update_job(a.job,status='succeeded',result=result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except Exception as exc:
        traceback.print_exc()
        store.update_job(a.job,status='failed',error=str(exc))
        sys.exit(1)


if __name__=='__main__':main()
