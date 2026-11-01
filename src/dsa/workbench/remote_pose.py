"""Run only detection and body pose in a new, isolated inference directory."""
import json
from pathlib import Path
import sys

from dsa.ingest.camera_dataset import detect, pose

root=Path(sys.argv[1]).resolve()
base=(Path.cwd()/'data/workbench-incoming').resolve()
if not root.is_relative_to(base) or root==base:
    raise ValueError('Remote imports must stay inside workbench-incoming')
detect(root,batch=4)
manifest=json.loads((root/'manifest.json').read_text())
if manifest['clips'][0].get('detection_status')!='complete':
    raise RuntimeError(manifest['clips'][0].get('exclusion_reason','Detection failed'))
pose(root,pose_batch=16)
manifest=json.loads((root/'manifest.json').read_text())
if manifest['clips'][0].get('pose_status')!='complete':
    raise RuntimeError(manifest['clips'][0].get('exclusion_reason','Pose failed'))
