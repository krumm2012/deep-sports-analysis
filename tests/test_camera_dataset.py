import io
from zipfile import ZipFile, ZipInfo
import numpy as np
import pytest
from dsa.ingest.camera_dataset import clip_seconds, safe_video_members, choose_subject


def test_zip_excludes_mac_metadata_and_orders_chronologically():
    b=io.BytesIO()
    with ZipFile(b,'w') as z:
        for name in ['camera_05/18.30.mp4','camera_05/00.20.mp4','__MACOSX/camera_05/._00.20.mp4','camera_05/.DS_Store']:
            z.writestr(name,b'data')
    with ZipFile(b) as z:
        assert [i.filename for i in safe_video_members(z)]==['camera_05/00.20.mp4','camera_05/18.30.mp4']
    assert clip_seconds('18.30.mp4')==1110


@pytest.mark.parametrize('name',['../escape.mp4','/tmp/escape.mp4','camera_05/../escape.mp4','camera_05\\escape.mp4'])
def test_zip_rejects_unsafe_names(name):
    b=io.BytesIO()
    with ZipFile(b,'w') as z:z.writestr(name,b'data')
    with ZipFile(b) as z:
        with pytest.raises(ValueError,match='Unsafe'):safe_video_members(z)


def test_zip_rejects_symlink():
    b=io.BytesIO();info=ZipInfo('camera_05/x.mp4');info.external_attr=(0o120777<<16)
    with ZipFile(b,'w') as z:z.writestr(info,b'/tmp/target')
    with ZipFile(b) as z:
        with pytest.raises(ValueError,match='Unsafe'):safe_video_members(z)


def test_subject_selection_keeps_static_drill_and_rejects_background():
    boxes=[np.array([[700,100,900,600,.9],[100,20,200,100,.99]]) for _ in range(30)]
    ids=[np.array([1,2]) for _ in boxes]
    wanted, chosen, ranking=choose_subject(boxes,ids,30,1280,720)
    assert chosen['person']==1 and len(wanted)==30
    assert all(j==0 for j in wanted.values())
    ids=[np.array([1 if i<20 else -1,2]) for i in range(30)]
    with pytest.raises(ValueError,match='80%'):choose_subject(boxes,ids,30,1280,720)


def test_standardization_uses_original_pts_and_25fps(monkeypatch,tmp_path):
    from dsa.ingest.camera_dataset import standardize
    from pathlib import Path
    import json
    commands=[]
    def run(cmd,check):
        commands.append(cmd);Path(cmd[-1]).touch()
    class Cap:
        def __init__(self,p):pass
        def get(self,key):
            import cv2
            return 25 if key==cv2.CAP_PROP_FPS else 50
        def release(self):pass
    monkeypatch.setattr('dsa.ingest.camera_dataset.subprocess.run',run)
    monkeypatch.setattr('dsa.ingest.camera_dataset.cv2.VideoCapture',Cap)
    c={'id':'x','raw':'raw/x.mp4','sha256':'hash','archive_member':'camera_05/x.mp4',
       'original_frames':50,'original_pts_seconds':(np.arange(50)/25).tolist(),
       'original_resolution':[2560,1440],'duration_seconds':2}
    result=standardize(tmp_path,c)
    assert result['fps']==25 and result['frames']==50
    assert 'fps=25,scale=1280:720' in commands[0]
    p=json.loads((tmp_path/'clips/x/source-provenance.json').read_text())
    assert p['processed_fps']==25
    assert p['nearest_pts_mapping_duplicated_output_frames']==0
    assert p['nearest_pts_mapping_unrepresented_original_frames']==0
