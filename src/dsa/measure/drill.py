"""Export a tracked drill player's 2D skeleton for within-clip MoSeq experiments.

Court homography and F3ED broadcast labels are not required for this body-motion input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from dsa.scene import clean, tracks

BODY = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
NAMES = ["nose", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
         "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_bigtoe",
         "l_smalltoe", "l_heel", "r_bigtoe", "r_smalltoe", "r_heel"]


def normalize(xy: np.ndarray) -> tuple[np.ndarray, float]:
    hip, shoulder = xy[:, 7:9].mean(1), xy[:, 1:3].mean(1)
    torso = float(np.nanmedian(np.linalg.norm(shoulder - hip, axis=1)))
    if not np.isfinite(torso) or torso <= 0:
        raise ValueError("Cannot establish a positive torso scale")
    return (xy - np.nanmedian(hip, axis=0)) / torso * 100, torso


def export(folder: Path, roi=(.5, .1, .98, .85)) -> Path:
    z = np.load(folder / "raw.npz", allow_pickle=False)
    fps, w, h = float(z["fps"]), int(z["width"]), int(z["height"])
    n = len(z["n"])
    offsets = np.cumsum(z["n"])[:-1]
    boxes, ids, kps = (np.split(z[key], offsets) for key in ("boxes", "track", "kp"))
    people = tracks.people(boxes, ids, fps)
    x0, y0, x1, y1 = np.array(roi) * [w, h, w, h]
    ranked = []
    for person, rec in people["people"].items():
        valid = []
        for f in rec["frames"]:
            js = [j for j, track in enumerate(ids[f]) if track >= 0 and people["person_of"][int(track)] == person]
            if not js:
                continue
            b = boxes[f][max(js, key=lambda j: boxes[f][j, 4])]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if x0 <= cx <= x1 and y0 <= cy <= y1 and b[3] - b[1] >= .2 * h:
                valid.append(float(b[3] - b[1]))
        ranked.append({"person": int(person), "roi_frames": len(valid),
                       "median_height": float(np.median(valid)) if valid else 0,
                       "score": float(len(valid) * np.median(valid)) if valid else 0})
    if not ranked or max(r["score"] for r in ranked) == 0:
        raise ValueError("No persistent full-body player found inside ROI")
    chosen = max(ranked, key=lambda r: r["score"])
    if chosen["roi_frames"] / n < .8:
        raise ValueError("Main player is not continuously tracked through at least 80% of the clip")
    pose = np.full((n, 65, 3), np.nan)
    height = np.full(n, np.nan)
    selected_boxes = np.full((n, 5), np.nan)
    for f in range(n):
        js = [j for j, track in enumerate(ids[f]) if track >= 0 and people["person_of"][int(track)] == chosen["person"]]
        if js:
            j = max(js, key=lambda j: boxes[f][j, 4])
            pose[f], selected_boxes[f] = kps[f][j], boxes[f][j]
            height[f] = boxes[f][j, 3] - boxes[f][j, 1]
    observed = np.isfinite(pose[:, BODY, :2]).all(2) & (pose[:, BODY, 2] >= .3)
    prepared = clean.joints(pose, height)[:, BODY]
    xy = prepared[:, :, :2].copy()
    remaining = ~np.isfinite(xy).all(2)
    if observed.mean() < .9:
        raise ValueError(f"Raw body-keypoint coverage too low: {observed.mean():.1%}")
    for j in range(19):
        for c in range(2):
            good = np.isfinite(xy[:, j, c])
            if good.sum() < 2:
                raise ValueError(f"Insufficient observations for {NAMES[j]}")
            xy[:, j, c] = np.interp(np.arange(n), np.flatnonzero(good), xy[good, j, c])
    conf = np.clip(np.nan_to_num(pose[:, BODY, 2]) / 3, 0, 1)
    conf[~observed | remaining] = 0  # Imputed points must not count as observations.
    coords, torso = normalize(xy)
    provenance = json.loads((folder / "source-provenance.json").read_text())
    times = np.arange(n) / fps
    original_pts = np.array(provenance["original_pts_seconds"])
    original_index = np.abs(times[:, None] - original_pts[None]).argmin(1)
    quality = {"subject": chosen, "candidate_tracks": ranked, "roi_fraction": list(roi),
               "frames": n, "fps": fps, "original_frame_count": len(original_pts),
               "nearest_original_frames_represented": int(len(np.unique(original_index))),
               "raw_observed_keypoint_fraction": float(observed.mean()),
               "remaining_missing_fraction_before_final_imputation": float(remaining.mean()),
               "zero_confidence_fraction": float((conf == 0).mean()), "median_torso_pixels": torso,
               "normalization": "subtract clip median hip center; divide clip median torso; multiply by 100",
               "confidence": "clip(raw RTMW score/3,0,1); all imputed/low-confidence points set to zero",
               "source": provenance,
               "limitations": [f"One {n / fps:.3f}-second clip from one drill scene; training-set segmentation is not generalization.",
                               "Resampled duplicates are not independent new observations.",
                               "Body keypoints exclude the racket; no grip or 3D joint-angle measurement.",
                               "No full court is visible; no court homography or broadcast-trained stroke labels used."]}
    path = folder / "moseq-input.npz"
    np.savez_compressed(path, coordinates=coords.astype(np.float32), confidences=conf.astype(np.float32),
                        image_coordinates=xy.astype(np.float32), raw_observed=observed,
                        bodyparts=np.array(NAMES), fps=fps, frame_indices=np.arange(n),
                        source_original_frame_indices=original_index, selected_boxes=selected_boxes,
                        metadata=json.dumps(quality, ensure_ascii=False))
    (folder / "pose-quality.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2))
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--roi", nargs=4, type=float, default=[.5, .1, .98, .85])
    a = p.parse_args()
    if not (0 <= a.roi[0] < a.roi[2] <= 1 and 0 <= a.roi[1] < a.roi[3] <= 1):
        p.error("ROI must be normalized xmin ymin xmax ymax")
    print(export(a.folder, a.roi))


if __name__ == "__main__":
    main()
