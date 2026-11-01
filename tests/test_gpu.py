"""The GPU steps' laptop-side logic, with stand-ins for the models: nothing runs a model here."""
import numpy as np
import pytest

from dsa.gpu import cuts, strokes


def _crowded_video(monkeypatch, *, tv=False):
    """Three frames each contain more people than a small pose forward permits."""
    from dsa import video
    from dsa.gpu import process as P
    from dsa.gpu import models as M
    from dsa.scene import court, tracks

    frames = [np.full((16, 16, 3), i, np.uint8) for i in range(3)]
    monkeypatch.setattr(video, "frames", lambda *a, **k: iter(frames))
    monkeypatch.setattr(video, "info", lambda _: (30, 16, 16))
    monkeypatch.setattr(tracks, "track", lambda boxes, *a, **k: [np.arange(len(b)) for b in boxes])
    monkeypatch.setattr(tracks, "people", lambda *a, **k: {
        "people": {j: {"still": False} for j in range(5)}, "person_of": {j: j for j in range(5)}})
    monkeypatch.setattr(court, "check_tv", lambda *a: np.zeros((14, 2)))
    monkeypatch.setattr(court, "homography", lambda _: None)
    batches = []

    class Det:
        def people(self, fs):
            return [np.array([[j, 0, j + 1, 10, .9] for j in range(5)], np.float32) for _ in fs]

    class Ball:
        small = staticmethod(lambda f: f)

        def locate(self, triplets, *a):
            return np.zeros((len(triplets), 4))

    class Pose:
        def prepare(self, frame, box):
            return int(frame[0, 0, 0]) * 10 + int(box[0])

        def run(self, crops):
            assert len(crops) <= 3, "oversized pose forward can exhaust GPU memory"
            batches.append(list(crops))
            return [np.full((len(M.KP), 3), v, np.float32) for v in crops]

    models = {"det": Det(), "ball": Ball(), "pose": Pose()}
    if tv:
        models["tcd"] = object()
    return P, M, models, batches, Det, Ball, Pose


@pytest.mark.parametrize("tv", [False, True])
def test_pose_forwards_have_a_hard_limit_and_keep_frame_box_order(tmp_path, monkeypatch, tv):
    P, _, models, batches, *_ = _crowded_video(monkeypatch, tv=tv)
    originals = dict(models)
    out = tmp_path / "raw.npz"
    P.process("crowded.mp4", str(out), tv=tv, batch=2, pose_batch=3, models=models)
    expected = [i * 10 + j for i in range(3) for j in range(5)]
    assert [v for batch in batches for v in batch] == expected
    assert np.load(out)["kp"][:, 0, 0].tolist() == expected
    assert models == originals  # A caller's shared models remain reusable.


def test_phone_releases_owned_pass1_models_before_constructing_pose(tmp_path, monkeypatch):
    import sys
    import weakref
    from types import SimpleNamespace

    P, M, _, _, Det, Ball, Pose = _crowded_video(monkeypatch)
    refs, releases = [], []

    def load(*a):
        det, ball = Det(), Ball()
        refs.extend([weakref.ref(det), weakref.ref(ball)])
        return {"det": det, "ball": ball}

    class CheckedPose(Pose):
        def __init__(self, device):
            assert all(r() is None for r in refs), "pass1 models still occupy GPU memory"
            assert releases == [True], "unused CUDA cache must be released before ORT allocation"

    monkeypatch.setattr(P, "load_models", load)
    monkeypatch.setattr(M, "Pose", CheckedPose)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: releases.append(True))))
    P.process("crowded.mp4", str(tmp_path / "raw.npz"), pose_batch=3)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_pose_limit_is_rejected_before_video_or_models_are_opened(size):
    from dsa.gpu.process import process

    with pytest.raises(ValueError, match="pose_batch"):
        process("missing.mp4", "unused.npz", pose_batch=size)


@pytest.mark.parametrize("pose_still", [False, True])
def test_stationary_drill_players_can_be_posed_without_changing_default(tmp_path, monkeypatch, pose_still):
    from dsa.scene import tracks

    P, _, models, batches, *_ = _crowded_video(monkeypatch)
    monkeypatch.setattr(tracks, "people", lambda *a, **k: {
        "people": {j: {"still": True} for j in range(5)}, "person_of": {j: j for j in range(5)}})
    result = P.process("drill.mp4", str(tmp_path / "raw.npz"), models=models,
                       pose_batch=3, pose_still=pose_still)
    assert result["posed"] == (15 if pose_still else 0)
    assert sum(map(len, batches)) == result["posed"]


def test_pose_rejects_silent_cpu_fallback(monkeypatch):
    import sys
    from types import SimpleNamespace

    import pytest

    from dsa.gpu.models import Pose

    session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(preload_dlls=lambda: None))
    monkeypatch.setitem(sys.modules, "rtmlib", SimpleNamespace(RTMPose=lambda *a, **k: SimpleNamespace(session=session)))
    with pytest.raises(RuntimeError, match="could not activate CUDA"):
        Pose("cuda")


def test_cuts_merge_short_views():
    flags = [False] * 10 + [True] * 2 + [False] * 5 + [True] * 20
    assert cuts.runs(flags, 6) == [[False, 0, 17], [True, 17, 37]]
    assert cuts.runs([True] * 3 + [False] * 20, 6) == [[False, 0, 23]]


def test_kept_pieces_join_adjacent_shots():
    shots = {"shots": [{"label": "keep", "start_frame": 0, "end_frame": 30},
                       {"label": "keep", "start_frame": 30, "end_frame": 50},                # joins the first
                       {"label": "discard", "start_frame": 50, "end_frame": 80},
                       {"label": "keep", "start_frame": 80, "end_frame": 200}]}
    assert cuts.kept_pieces(shots) == [[0, 50], [80, 200]]


def test_strokes_decode_maps_back_to_the_source():
    hit = np.tile([1.0, 0.0], (20, 1)).astype(np.float32)
    hit[7] = [0.1, 0.9]                                                                  # one hit, 2nd piece bin 7
    el = np.zeros((20, 29), np.float32)
    el[7, [1, 3, 7, 9, 10, 19, 25]] = 1                                                  # far middle stroke bh gs CC in
    quiet = np.tile([1.0, 0.0], (5, 1)).astype(np.float32)
    rows = strokes.decode({"hit": [quiet, hit], "elements": [np.zeros((5, 29), np.float32), el], "stride": 2},
                          {"fps": 25.0, "pieces": [(0, 10), (100, 140)]})
    assert [(r["frame"], r["source_frame"], r["side"], r["shot"], r["hand"], r["technique"], r["direction"],
             r["outcome"]) for r in rows] == [(24, 114, "far", "stroke", "bh", "gs", "CC", "in")]


def test_fine_windows_pad_hits_join_overlaps_and_stay_in_their_piece():
    from dsa.gpu.fine import windows

    hits = [{"side": "near", "source_frame": f} for f in (10, 14, 40, 101)] + [{"side": "far", "source_frame": 60}]
    pieces = [[0, 50], [100, 200]]                                   # source frames; upload at twice the rate
    assert windows(hits, pieces, ratio=2.0, pad=5) == [[15, 34], [75, 86], [200, 208]]
    #                       10 and 14 joined; 101 starts at its piece (frame 200), not 5 before; far hits left out


def test_frame_loop_keeps_frame_order_across_chunks_and_cuts(tmp_path, monkeypatch):
    """process() with stand-in models on a made-up video: frames decoded in chunks, the next chunk's ball inputs
    prepared while this one is detected, must still line up with their frames; the ball's neighbours stop at
    cuts."""
    import cv2

    from dsa.gpu import process as P
    from dsa.gpu.models import KP

    video, n = str(tmp_path / "v.mp4"), 11
    w = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for i in range(n):
        w.write(np.full((48, 64, 3), 20 * i, np.uint8))              # frame i is grey level 20 i
    w.release()
    index = lambda f: int(round(f.mean() / 20))                       # noqa: E731

    class Det:
        def people(self, fs):
            return [np.array([[index(f), 0, index(f) + 10, 40, 0.9]], np.float32) for f in fs]

    class Ball:
        small = staticmethod(index)

        def locate(self, triplets, W, H):
            return [[1, *t] for t in triplets]                       # (found, previous, this, next frame)

    class Pose:
        def prepare(self, f, box):
            return index(f)

        def run(self, crops):
            return [np.zeros((len(KP), 3)) for _ in crops]

    monkeypatch.setattr(P, "CHUNK", 4)
    r = P.process(video, str(tmp_path / "raw.npz"), batch=3, breaks=[6], models={"det": Det(), "ball": Ball(),
                                                                                    "pose": Pose()})
    z = np.load(tmp_path / "raw.npz")
    assert r["frames"] == n and z["boxes"][:, 0].tolist() == list(range(n))
    assert z["ball"][:, 1:].tolist() == [[max(i - 1, 0) if i != 6 else 6, i, min(i + 1, n - 1) if i != 5 else 5]
                                         for i in range(n)]
