"""The optional backend uses an existing SSH alias; no credentials are copied."""
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile


def read_backend(path):
    d=json.loads(Path(path).read_text())
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',d['ssh_host']):raise ValueError('Use an existing SSH alias')
    if not d['project_directory'].startswith('/') or '\n' in d['project_directory']:raise ValueError('Invalid backend directory')
    if not d['python'] or '\n' in d['python']:raise ValueError('Invalid backend interpreter')
    return d


def ssh_command(settings, command):
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',settings['ssh_host'],command]


def check_backend(path):
    settings=read_backend(path)
    p=subprocess.run(ssh_command(settings,"nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),capture_output=True,text=True,timeout=15)
    return {'configured':True,'online':p.returncode==0,'host':settings['ssh_host'],
            'device':p.stdout.strip() if p.returncode==0 else None,
            'error':p.stderr.strip()[-500:] if p.returncode else None}


def safe_extract(file, root):
    with tarfile.open(fileobj=file,mode='r:gz') as tar:
        members=tar.getmembers()
        if sum(m.size for m in members)>500*1024**2:raise ValueError('Backend output too large')
        for m in members:
            p=PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or not m.isfile() or '\\' in m.name:
                raise ValueError('Unsafe backend result')
            if m.name not in ['manifest.json','clips/video/moseq-input.npz','clips/video/pose-quality.json']:
                raise ValueError('Unexpected backend result')
        for m in members:
            target=root/m.name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(tar.extractfile(m).read())
