"""Run step 2 on an existing NVIDIA machine, without Modal or GCP.

    python -m dsa.gpu.local <project>/<video> --seconds 60
    python -m dsa.gpu.local <project>/<video> --cuts
"""
from __future__ import annotations

import argparse
import json
import numpy as np

from dsa.gpu.process import plan, process
from dsa.ingest.projects import PROJECTS
from dsa.paths import CODE, F3ED, TCD_WEIGHTS, WASB, video_dir


def run(spec: str, seconds: float = 0, cuts: bool = False, tv: bool = False) -> dict:
    import torch

    if seconds < 0:
        raise ValueError("seconds must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and NVIDIA GPU are required")
    d = video_dir(spec)
    if not (d / "source.mp4").is_file():
        raise FileNotFoundError(f"Missing {d / 'source.mp4'}; ingest a video first")
    if cuts:
        from dsa.gpu.cuts import run as cut
        if not TCD_WEIGHTS.exists():
            raise FileNotFoundError(TCD_WEIGHTS)
        cut(spec, "cuda")
    name, lengths, fps = plan(spec, seconds)
    if not lengths or sum(lengths) <= 0 or fps <= 0:
        raise ValueError("The input video has no readable frames")
    is_tv = tv or name == "kept.mp4"
    p = PROJECTS[spec.split("/")[0]]
    strokes = is_tv or p.get("strokes", False)
    required = [WASB / "src/configs/model/wasb.yaml", WASB / "pretrained_weights/wasb_tennis_best.pth.tar"]
    if is_tv:
        required += [CODE / "tennis_court_detector/tracknet.py", TCD_WEIGHTS]
    if strokes:
        required += [CODE / "f3set/train_f3set_f3ed.py", F3ED / "config.json", F3ED / "checkpoint_049.pt"]
    missing = [str(f) for f in required if not f.exists()]
    if missing:
        raise FileNotFoundError("Provision third-party model code/weights first:\n" + "\n".join(missing))
    out = d / "test" if seconds else d
    out.mkdir(parents=True, exist_ok=True)
    if strokes and not (out / "strokes/strokes.json").exists():
        from dsa.gpu.strokes import decode, f3ed
        scores = f3ed(str((d / name).resolve()), lengths, fps)
        sd = out / "strokes"
        sd.mkdir(exist_ok=True)
        np.savez_compressed(sd / "scores.npz", stride=scores["stride"],
                            **{f"hit_{k}": h for k, h in enumerate(scores["hit"])},
                            **{f"elements_{k}": e for k, e in enumerate(scores["elements"])})
        source = json.loads((d / "kept.json").read_text()) if name == "kept.mp4" else {"pieces": [[0, lengths[0]]]}
        pieces = [[a, a + n] for (a, _), n in zip(source["pieces"], lengths)]
        hits = decode(scores, {"fps": fps, "pieces": pieces})
        (sd / "strokes.json").write_text(json.dumps({"fps": fps, "frames": sum(lengths), "hits": hits}, indent=1))
    if (out / "raw.npz").exists():
        return {"status": "already done", "output": str(out)}
    return process(str(d / name), str(out / "raw.npz"), tv=is_tv, limit=sum(lengths),
                   breaks=np.cumsum(lengths)[:-1].tolist(), **p.get("gpu", {}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--cuts", action="store_true", help="First select live play on this GPU (TV)")
    ap.add_argument("--tv", action="store_true", help="Broadcast video without kept.json")
    a = ap.parse_args()
    print(json.dumps(run(a.video, a.seconds, a.cuts, a.tv), indent=2))


if __name__ == "__main__":
    main()
