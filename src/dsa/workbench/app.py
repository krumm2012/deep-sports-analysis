"""Loopback-only workbench API and UI for fixed-camera phase projects."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Thread
from urllib.parse import parse_qs, unquote, urlsplit

from dsa.pages.serve import Ranges
from dsa.workbench.store import Store, Conflict, atomic_json, progress_hash


class Workbench:
    def __init__(self,data,backend_config=None):
        self.store=Store(data)
        self.backend_config=Path(backend_config).resolve() if backend_config else None
        self.recover()

    def recover(self):
        # Detached workers survive a server restart. Only dead jobs become retryable.
        with self.store.connect() as db:
            rows=db.execute("SELECT id,pid,status FROM jobs WHERE status IN ('queued','running')").fetchall()
        for row in rows:
            try:
                if not row['pid']:raise ProcessLookupError()
                os.kill(row['pid'],0)
            except ProcessLookupError:
                self.store.update_job(row['id'],status='interrupted',error='Worker stopped; the original inputs remain available for retry')

    def launch(self,job):
        folder=self.store.directory/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=True)
        environment=os.environ.copy()
        source=str(Path(__file__).resolve().parents[2])
        environment['PYTHONPATH']=source+os.pathsep+environment.get('PYTHONPATH','')
        try:
            with (folder/'worker.log').open('ab') as log:
                process=subprocess.Popen([sys.executable,'-u','-m','dsa.workbench.worker','--data',str(self.store.data),'--job',job['id']],
                    stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
            self.store.update_job(job['id'],pid=process.pid)
            def monitor():
                code=process.wait()
                state=self.store.job(job['id'])
                if state['status'] in ['queued','running']:
                    self.store.update_job(job['id'],status='failed',error=f'Worker exited before completing (exit {code}); inspect the task log')
            Thread(target=monitor,daemon=True).start()
        except Exception as e:
            self.store.update_job(job['id'],status='failed',error=str(e));raise
        return self.store.job(job['id'])

    def summary(self,project):
        store=self.store;root=store.project(project);current=store.current(project)
        manifest=json.loads((root/'manifest.json').read_text());annotation=store.annotations(project)
        excluded=set(current.get('excluded_clips',[]))
        clips=[{'id':c['id'],'split':c['split'],'frames':c['frames'],'fps':c['fps']} for c in manifest['clips']
               if c['split'] in ['train','validation'] and c['id'] not in excluded]
        return {'id':project,'current':current,'clips':clips,'models':store.models(project),
                'completed':len(annotation['payload']['completed_tasks']),
                'total':len(annotation['payload']['annotation_progress']['tasks']),
                'annotation_revision':annotation['revision'],'saved_at':annotation['saved_at'],
                'jobs':store.jobs(project),'editor_url':'/workbench/editor?project='+project,
                'report_url':'/projects/'+project+'/'+current['report']}

    def train(self,project):
        snapshot=self.store.annotations(project)
        q=self.store.queue(project)
        if len(snapshot['payload']['completed_tasks'])!=len(q['tasks']):
            raise ValueError('Confirm every active training clip before training')
        job=self.store.new_job(project,'train',{'annotation_revision':snapshot['revision'],
            'progress_hash':snapshot['progress_hash']})
        atomic_json(self.store.directory/'jobs'/job['id']/'annotation.json',snapshot['payload'])
        return self.launch(job)

    def retry(self,identifier):
        old=self.store.job(identifier)
        if old['status'] not in ['failed','interrupted']:raise ValueError('Only failed or interrupted jobs may be retried')
        request=dict(old['request'])
        if old['action']=='import_video':request['source_import_id']=request.get('source_import_id',old['id'])
        job=self.store.new_job(old['project'],old['action'],request)
        if old['action']=='train':
            source=self.store.directory/'jobs'/old['id']/'annotation.json'
            atomic_json(self.store.directory/'jobs'/job['id']/'annotation.json',json.loads(source.read_text()))
        return self.launch(job)

    def predict(self,project,request):
        from dsa.measure.semantic_inference import predict
        root=self.store.project(project);model,_=self.store.model(project,request.get('version'))
        if request.get('import_id'):
            job=self.store.job(request['import_id'])
            if job['project']!=project or job['action']!='import_video' or job['status']!='succeeded':raise ValueError('Unknown prepared import')
            source=self.store.directory/'imports'/job['id']/'clips/video/moseq-input.npz'
        else:
            clips=json.loads((root/'manifest.json').read_text())['clips']
            clip=next((c for c in clips if c['id']==request.get('clip_id') and c['split'] in ['train','validation'] and c['id'] not in self.store.current(project).get('excluded_clips',[])),None)
            if not clip:raise ValueError('Choose an active training or validation clip')
            source=self.store.inside(root,clip['input'])
        return predict(model,source,request.get('stroke_type','无法判断'))


class Handler(Ranges):
    server_version='DSAWorkbench/1'

    @property
    def app(self):return self.server.workbench

    def json(self,value,status=200):
        body=json.dumps(value,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)

    def trusted(self,mutating=False):
        port=self.server.server_port
        hosts={f'127.0.0.1:{port}',f'localhost:{port}'}
        if self.headers.get('Host') not in hosts:raise PermissionError('Invalid workbench host')
        origin=self.headers.get('Origin')
        if origin and origin not in {'http://'+host for host in hosts}:raise PermissionError('Cross-origin requests are not allowed')
        if mutating and self.headers.get('X-Workbench-Request')!='1':raise PermissionError('Use the workbench request header')

    def body(self):
        length=int(self.headers.get('Content-Length','0'))
        if not 0<length<=10*1024**2:raise ValueError('Invalid JSON body length')
        value=json.loads(self.rfile.read(length))
        if not isinstance(value,dict):raise ValueError('Expected a JSON object')
        return value

    def dispatch(self,method):
        try:
            self.trusted(method!='GET')
            self.route(method)
        except Conflict as e:self.json({'error':str(e)},409)
        except PermissionError as e:self.json({'error':str(e)},403)
        except (ValueError,KeyError,FileNotFoundError,TypeError) as e:self.json({'error':str(e)},400)
        except Exception as e:
            import traceback;traceback.print_exc()
            self.json({'error':str(e)},500)

    def do_GET(self):self.dispatch('GET')
    def do_POST(self):self.dispatch('POST')
    def do_PUT(self):self.dispatch('PUT')

    def translate_path(self,path):
        if hasattr(self,'file_override'):return str(self.file_override)
        parts=[unquote(p) for p in urlsplit(path).path.split('/') if p]
        root=self.app.store.data
        p=root.joinpath(*parts).resolve()
        if '.workbench' in parts or not p.is_relative_to(root):raise PermissionError('Private workbench state is not a static asset')
        return str(p)

    def send_file(self,path):
        self.file_override=path
        return super().do_GET()

    def route(self,method):
        url=urlsplit(self.path);path=url.path;args=parse_qs(url.query);app=self.app;store=app.store
        if method=='GET' and path in ['/','/workbench/']:
            return self.send_file(Path(__file__).with_name('index.html'))
        if method=='GET' and path=='/workbench/app.js':return self.send_file(Path(__file__).with_name('app.js'))
        if method=='GET' and path=='/api/health':
            import platform
            return self.json({'status':'ok','service':'dsa-workbench','platform':platform.machine(),'free_gib':round(shutil.disk_usage(store.data).free/1024**3,1),
                'backend_configured':bool(app.backend_config),'data_directory':str(store.data)})
        if method=='GET' and path=='/api/projects':
            projects=[]
            for p in (store.data/'projects').glob('*/current-semantic.json'):
                try:projects.append(app.summary(p.parent.name))
                except (ValueError,FileNotFoundError,KeyError):continue
            return self.json(projects)
        if method=='GET' and path=='/workbench/editor':
            project=args['project'][0];q=store.queue(project);saved=store.annotations(project)
            from html import escape
            root=store.project(project);editor=store.inside(root,store.current(project)['editor'])
            q['initial_progress']=saved['payload']['annotation_progress']
            q['storage_prefix']='workbench-'+project+'-'
            q['server_sync']={'url':f'/api/projects/{project}/annotations','revision':saved['revision']}
            template=Path(__file__).parents[1].joinpath('pages/semantic_labeling.html').read_text()
            template=template.replace('/*__TIMELINE__*/',Path(__file__).parents[1].joinpath('pages/semantic_timeline.js').read_text())
            template=template.replace('<head>','<head><base href="'+escape('/projects/'+project+'/'+store.current(project)['editor'])+'">',1)
            proposals=editor/'auto-segments.json'
            suggestions=json.loads(proposals.read_text()) if proposals.exists() else {}
            html=template.replace('__TITLE__',escape(q['title'])).replace('__QUEUE__',json.dumps(q,ensure_ascii=False).replace('<','\\u003c')).replace('__AUTO__',json.dumps(suggestions,ensure_ascii=False).replace('<','\\u003c'))
            body=html.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
        if path.startswith('/api/jobs/'):
            parts=path.strip('/').split('/');job=store.job(parts[2])
            if method=='POST' and len(parts)==4 and parts[3]=='retry':return self.json(app.retry(job['id']),202)
            if method=='GET' and len(parts)==4 and parts[3]=='log':
                p=store.directory/'jobs'/job['id']/'worker.log'
                return self.json({'log':p.read_text(errors='replace')[-12000:] if p.exists() else ''})
            if method=='GET' and len(parts)==3:return self.json(job)
        if path.startswith('/api/imports/') and method=='GET':
            parts=path.strip('/').split('/');job=store.job(parts[2])
            if job['action']!='import_video':raise ValueError('Unknown import')
            if len(parts)==4 and parts[3]=='video':return self.send_file(store.directory/'imports'/job['id']/'clips/video/source.mp4')
        if path.startswith('/api/projects/'):
            parts=path.strip('/').split('/');project=parts[2];store.project(project)
            action=parts[3] if len(parts)>3 else ''
            if method=='GET' and not action:return self.json(app.summary(project))
            if action=='annotations':
                if method=='GET':return self.json(store.annotations(project))
                if method=='PUT':
                    b=self.body();saved=store.save(project,b['queue_id'],b['revision'],b['progress'])
                    return self.json({k:v for k,v in saved.items() if k!='payload'})
            if method=='GET' and action=='export':return self.json(store.annotations(project)['payload'])
            if method=='POST' and action=='train':return self.json(app.train(project),202)
            if method=='POST' and action=='predict':return self.json(app.predict(project,self.body()))
            if method=='POST' and action=='backend-check':
                if not app.backend_config:return self.json({'configured':False,'online':False})
                from dsa.workbench.backend import check_backend
                return self.json(check_backend(app.backend_config))
            if method=='POST' and action=='remote-pose':
                if not app.backend_config:raise ValueError('No 5090 backend configured')
                b=self.body();item=store.job(b['import_id'])
                if item['project']!=project or item['action']!='import_video' or item['status']!='succeeded':raise ValueError('Choose a prepared video import')
                return self.json(app.launch(store.new_job(project,'remote_pose',{'import_id':item['id'],'backend_config':str(app.backend_config)})),202)
            if method=='POST' and action=='imports':return self.upload(project,args)
        if method=='GET' and path.startswith('/projects/'):
            components=path.strip('/').split('/')
            # Reports must lead back to the one durable annotation source, including
            # reports for newly trained models. Never revive an old browser-only draft.
            if len(components)>=4 and components[2].startswith('semantic-pose-') and components[3]=='edit':
                project=components[1];store.project(project)
                self.send_response(302)
                location='/workbench/editor?project='+project
                if args.get('clip'):location+='&clip='+args['clip'][0]
                self.send_header('Location',location);self.end_headers();return
            return super().do_GET()
        self.json({'error':'Not found'},404)

    def upload(self,project,args):
        app=self.app;store=app.store
        filename=Path(args.get('filename',['video.mp4'])[0]).name
        if Path(filename).suffix.lower()!='.mp4':raise ValueError('Upload an MP4 fixed-camera clip')
        length=int(self.headers.get('Content-Length','0'))
        if not 0<length<=500*1024**2:raise ValueError('Upload size must be between 1 byte and 500MB')
        if shutil.disk_usage(store.data).free-length<2*1024**3:raise ValueError('Keep at least 2GiB free after upload')
        job=store.new_job(project,'import_video',{'filename':filename})
        target=store.directory/'imports'/job['id']/'raw/original.mp4'
        target.parent.mkdir(parents=True,exist_ok=True)
        try:
            with target.with_suffix('.part').open('wb') as f:
                left=length
                while left:
                    chunk=self.rfile.read(min(left,1024**2))
                    if not chunk:raise ValueError('Upload interrupted')
                    f.write(chunk);left-=len(chunk)
            target.with_suffix('.part').replace(target)
            return self.json(app.launch(job),202)
        except Exception as e:
            store.update_job(job['id'],status='failed',error=str(e));raise


def make_server(data,port=8777,backend_config=None):
    app=Workbench(data,backend_config)
    server=ThreadingHTTPServer(('127.0.0.1',port),partial(Handler,directory=str(app.store.data)))
    server.workbench=app
    return server


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('data'));p.add_argument('--port',type=int,default=8777);p.add_argument('--backend-config',type=Path);a=p.parse_args()
    server=make_server(a.data,a.port,a.backend_config)
    print(f'Workbench: http://127.0.0.1:{server.server_port}/',flush=True)
    try:server.serve_forever()
    finally:server.server_close()


if __name__=='__main__':main()
