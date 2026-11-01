"""Install and manage the per-user Mac workbench LaunchAgent."""
import argparse
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

LABEL='com.deepsports.workbench'


def bootstrap(domain, target):
    # bootout returns before launchd releases the registration. Retry that brief
    # transition rather than reporting a failed deployment after a normal restart.
    for _ in range(20):
        result=subprocess.run(['launchctl','bootstrap',domain,str(target)],capture_output=True,text=True)
        if result.returncode==0:return
        time.sleep(.2)
    raise RuntimeError(result.stderr.strip() or 'LaunchAgent bootstrap failed')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','restart','stop','status']);p.add_argument('--project',type=Path,required=True);p.add_argument('--data',type=Path);p.add_argument('--port',type=int,default=8777);a=p.parse_args()
    source_root=a.project.resolve();source_data=(a.data or source_root/'data').resolve()
    root=Path.home()/'Library/Application Support/DeepSportsWorkbench'
    data=root/'data';state=data/'.workbench';state.mkdir(parents=True,exist_ok=True)
    domain=f'gui/{os.getuid()}';service=domain+'/'+LABEL
    target=Path.home()/'Library/LaunchAgents'/f'{LABEL}.plist'
    if a.action=='status':
        with urlopen(f'http://127.0.0.1:{a.port}/api/health',timeout=3) as response:print(response.read().decode())
        return
    existing=subprocess.run(['launchctl','print',service],capture_output=True).returncode==0
    if a.action=='stop':
        if existing:subprocess.run(['launchctl','bootout',service],check=True)
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    # launchd cannot use a protected Documents working directory. Deploy a runtime
    # copy in Application Support. Existing runtime annotations are never replaced.
    shutil.copytree(source_root/'src',root/'src',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.egg-info'))
    runtime_python=root/'.venv-workbench/bin/python'
    if not runtime_python.exists():
        environment='.venv-workbench' if (source_root/'.venv-workbench').exists() else '.venv'
        if not (root/environment).exists():
            shutil.copytree(source_root/environment,root/environment,symlinks=True)
        runtime_python=root/environment/'bin/python'
    (root/'config').mkdir(exist_ok=True)
    shutil.copyfile(source_root/'config/workbench-backend.json',root/'config/workbench-backend.json')
    def retain_source(src,dst):
        if not Path(dst).exists():
            try:os.link(src,dst)
            except OSError:shutil.copy2(src,dst)
        return dst
    for current in (source_data/'projects').glob('*/current-semantic.json'):
        shutil.copytree(current.parent,data/'projects'/current.parent.name,dirs_exist_ok=True,copy_function=retain_source)
    (root/'deployment.json').write_text(json.dumps({'source_project':str(source_root),'source_data':str(source_data),'runtime_data':str(data),'storage':'Existing source files linked read-only by convention; new outputs are separate files.'},indent=2))
    config={'Label':LABEL,'ProgramArguments':[str(runtime_python),'-u','-m','dsa.workbench.app','--data',str(data),'--port',str(a.port),'--backend-config',str(root/'config/workbench-backend.json')],
            'WorkingDirectory':str(root),'EnvironmentVariables':{'PYTHONPATH':str(root/'src'),'DSA_DATA':str(data),'PATH':'/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin'},
            'RunAtLoad':True,'KeepAlive':True,'ThrottleInterval':5,'AbandonProcessGroup':True,
            'StandardOutPath':str(state/'server.log'),'StandardErrorPath':str(state/'server.log')}
    target.write_bytes(plistlib.dumps(config))
    if not existing:bootstrap(domain,target)
    elif a.action=='restart':
        subprocess.run(['launchctl','bootout',service],check=True)
        bootstrap(domain,target)
    for _ in range(90):
        try:
            with urlopen(f'http://127.0.0.1:{a.port}/api/health',timeout=1) as response:
                if json.load(response).get('service')=='dsa-workbench':
                    print(f'Workbench ready: http://127.0.0.1:{a.port}/');return
        except Exception:pass
        time.sleep(.3)
    raise RuntimeError(f'Workbench did not become ready; inspect {state / "server.log"}')


if __name__=='__main__':main()
