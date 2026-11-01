"""Audit a multi-video MoSeq fit and frozen-parameter held-out inference.

Paths in the dataset manifest are relative to its directory. Model output is
``RUN/results/CLIP_ID.npz``; each clip retains its own temporal boundaries.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote

import cv2
import numpy as np

from dsa.pages.moseq_report import _color, _draw, compare_labels, run_statistics


def relative(path: Path, output: Path) -> str:
    return quote(os.path.relpath(path.resolve(), output.resolve()), safe="/")


def aggregate(records: list[dict], training_states: set[int] | None = None) -> dict:
    """Pool occupancy, but never count a transition across clip boundaries."""
    invalid_count = sum(not r["statistics"]["technical_output_valid"] for r in records)
    records = [r for r in records if r["statistics"]["technical_output_valid"]]
    frames = sum(r["statistics"]["frames"] for r in records)
    seconds = sum(r["statistics"]["duration_seconds"] for r in records)
    pieces = [s for r in records for s in r["statistics"]["segments"]]
    durations = np.array([s["seconds"] for s in pieces])
    states: dict[int, int] = {}
    for r in records:
        for f in r["statistics"]["frequencies"]:
            states[f["state"]] = states.get(f["state"], 0) + f["frames"]
    transitions = sum(r["statistics"]["transition_count"] for r in records)
    unseen = sorted(set(states) - training_states) if training_states is not None else []
    return {
        "clips": len(records), "frames": frames, "duration_seconds": seconds,
        "invalid_output_clips": invalid_count,
        "active_states": len(states), "state_frames": states,
        "state_fractions": {s: n / frames for s, n in states.items()} if frames else {},
        "segment_count": len(pieces), "transition_count": transitions,
        "transitions_per_second": transitions / seconds if seconds else None,
        "median_segment_seconds": float(np.median(durations)) if len(durations) else None,
        "tiny_segment_fraction_under_0_1_seconds": float((durations < .1 - 1e-9).mean()) if len(durations) else None,
        "single_state_clip_count": sum(r["statistics"]["single_state_collapse"] for r in records),
        "dominant_state_clip_count": sum(r["statistics"]["dominant_state_over_95_percent"] for r in records),
        "largest_state_fraction": max(states.values()) / frames if frames else None,
        "states_absent_from_training": unseen,
        "unseen_state_frame_fraction": sum(states[s] for s in unseen) / frames if frames else 0,
        "technical_output_valid": bool(records) and invalid_count == 0,
    }


def distribution_shift(train: dict, validation: dict) -> dict:
    """Occupancy shift is diagnostic, not an action-accuracy measurement."""
    if not train["frames"] or not validation["frames"]:
        return {"jensen_shannon_divergence_bits": None, "total_variation": None}
    states = sorted(set(train["state_frames"]) | set(validation["state_frames"]))
    p = np.array([train["state_frames"].get(s, 0) / train["frames"] for s in states])
    q = np.array([validation["state_frames"].get(s, 0) / validation["frames"] for s in states])
    midpoint = (p + q) / 2
    def kl(values):
        nonzero = values > 0
        return float(np.sum(values[nonzero] * np.log2(values[nonzero] / midpoint[nonzero])))
    return {"jensen_shannon_divergence_bits": (kl(p) + kl(q)) / 2,
            "total_variation": float(np.abs(p - q).sum() / 2),
            "interpretation": "状态占比差异；受动作分布和片段长度影响，不是准确率，也没有预设通过阈值"}


def compare_runs(left: dict, right: dict, split: str) -> dict:
    """Compare aligned clips with a global, permutation-invariant partition."""
    ids = sorted(set(left["records"]) & set(right["records"]))
    pairs = []
    a, b = [], []
    for clip_id in ids:
        l, r = left["records"][clip_id], right["records"][clip_id]
        if l["split"] != split or r["split"] != split:
            continue
        if not l["statistics"]["technical_output_valid"] or not r["statistics"]["technical_output_valid"]:
            continue
        if not np.array_equal(l["frame_indices"], r["frame_indices"]):
            raise ValueError(f"Cross-run frame alignment differs for {clip_id}")
        agreement = compare_labels(l["labels"], r["labels"])
        pairs.append({"id": clip_id, "frames": len(l["labels"]), **agreement})
        a.append(l["labels"]); b.append(r["labels"])
    pooled = compare_labels(np.concatenate(a), np.concatenate(b)) if pairs else None
    return {"left": left["name"], "right": right["name"], "split": split,
            "clips": len(pairs), "pooled": pooled, "per_clip": pairs,
            "median_clip_ARI": float(np.median([p["adjusted_rand_index"] for p in pairs])) if pairs else None,
            "median_clip_NMI": float(np.median([p["normalized_mutual_information"] for p in pairs])) if pairs else None,
            "degenerate_clip_count": sum(p["degenerate_comparison"] for p in pairs)}


def load_inputs(clip: dict, base: Path) -> dict:
    path = base / clip["input"]
    with np.load(path, allow_pickle=False) as z:
        inputs = {k: z[k].copy() for k in z.files}
    xy, conf = inputs["image_coordinates"], inputs["confidences"]
    n = len(inputs["coordinates"])
    if inputs["coordinates"].shape != (n, 19, 2) or xy.shape != (n, 19, 2) or conf.shape != (n, 19):
        raise ValueError("Expected 19 aligned 2D keypoints and confidence values")
    if not np.isfinite(inputs["coordinates"]).all() or not np.isfinite(xy).all() or not np.isfinite(conf).all() or (conf < 0).any() or (conf > 1).any():
        raise ValueError("Input coordinates/confidences are invalid")
    frames = inputs["frame_indices"]
    if frames.shape != (n,) or not np.array_equal(frames, frames.astype(np.int64)) or (frames < 0).any() or (np.diff(frames) <= 0).any():
        raise ValueError("Input frame indices must be unique increasing integers")
    fps = float(inputs["fps"])
    if not np.isfinite(fps) or fps <= 0 or n < 1:
        raise ValueError("Input frame rate or length is invalid")
    inputs["fps"] = fps
    inputs["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return inputs


def frozen_parameter_status(summary: dict) -> dict:
    expected = summary.get("trained_parameter_sha256")
    checks = summary.get("frozen_parameter_checks", {})
    return {split: bool(expected) and checks.get(split, {}).get("unchanged") is True
            and all(checks[split].get(field) == expected
                    for field in ("before_sha256", "after_sha256", "original_model_after_sha256"))
            for split in ("train", "validation")}


def load_run(folder: Path, clips: list[dict], inputs: dict[str, dict], output: Path) -> dict:
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    frozen = frozen_parameter_status(summary)
    records, failures = {}, []
    for clip in clips:
        clip_id, split = clip["id"], clip["split"]
        if split not in {"train", "validation"}:
            continue
        try:
            source = inputs[clip_id]
            filename = folder / "results" / f"{clip_id}.npz"
            with np.load(filename, allow_pickle=False) as z:
                labels = z["syllable"].copy()
                arrays = {k: z[k].copy() for k in ("latent_state", "centroid", "heading")}
                if "fps" in z and not np.isclose(float(z["fps"]), source["fps"]):
                    raise ValueError("Output/input fps differ")
                if not np.array_equal(z["frame_indices"], source["frame_indices"]):
                    raise ValueError("Output/input frame indices differ")
                if "bodyparts" in z and not np.array_equal(z["bodyparts"], source["bodyparts"]):
                    raise ValueError("Output/input bodypart order differs")
            stats = run_statistics(labels, source["fps"], len(source["coordinates"]), arrays)
            per_clip = summary.get("per_clip", {}).get(clip_id, {})
            if per_clip.get("input_sha256") and per_clip["input_sha256"] != source["sha256"]:
                raise ValueError("Training summary/input SHA256 differ")
            if summary.get("status") != "complete":
                stats["technical_output_valid"] = False
                stats["interpretation"] = "该运行未完整结束，不能判定模型有效"
            if not frozen[split]:
                stats["technical_output_valid"] = False
                stats["interpretation"] = "固定参数推断的SHA256校验未通过或未提供，不能判定该集合输出有效"
            records[clip_id] = {"id": clip_id, "split": split, "statistics": stats,
                                "labels": labels, "frame_indices": source["frame_indices"],
                                "results": relative(filename, output),
                                "maximum_absolute_heading": float(np.abs(arrays["heading"]).max()),
                                "zero_confidence_fraction": float((source["confidences"] == 0).mean())}
        except (KeyError, ValueError, OSError) as exc:
            failures.append({"id": clip_id, "split": split, "error": str(exc)})
    train = [r for r in records.values() if r["split"] == "train"]
    train_stats = aggregate(train)
    validation = aggregate([r for r in records.values() if r["split"] == "validation"], set(train_stats["state_frames"]))
    checkpoints = sorted(folder.rglob("*checkpoint*.h5"))
    return {"name": folder.name, "summary": summary,
            "summary_link": relative(summary_path, output), "records": records,
            "failures": failures, "aggregate": {"train": train_stats, "validation": validation},
            "train_validation_occupancy_shift": distribution_shift(train_stats, validation),
            "frozen_parameter_verification": frozen,
            "checkpoints": [relative(p, output) for p in checkpoints],
            "has_checkpoint": bool(checkpoints)}


def draw(frame: np.ndarray, source: dict, i: int, state: int, clip_id: str, split: str) -> np.ndarray:
    image = _draw(frame, source["image_coordinates"][i], source["confidences"][i], state, i / source["fps"])
    y = image.shape[0] - 48
    image[y:] = 20
    seconds = int(source["frame_indices"][i]) / source["fps"]
    caption = f"{clip_id} | state {state} | {seconds:.2f}s | {'held-out inference' if split == 'validation' else 'train reapplication'}"
    cv2.putText(image, caption, (12, y + 30), cv2.FONT_HERSHEY_SIMPLEX, .65, _color(state), 2, cv2.LINE_AA)
    return image


def overlay(video: Path, target: Path, record: dict, source: dict) -> None:
    cap = cv2.VideoCapture(str(video))
    w, h = (int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if min(w, h) <= 0:
        raise ValueError(f"Unreadable video {video}")
    process = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                                "-r", str(source["fps"]), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    lookup = {int(f): i for i, f in enumerate(source["frame_indices"])}
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n in lookup:
                i = lookup[n]
                frame = draw(frame, source, i, int(record["labels"][i]), record["id"], record["split"])
            process.stdin.write(frame.tobytes()); n += 1
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        if process.wait():
            raise RuntimeError(f"ffmpeg: {error}")
        if n <= max(lookup):
            raise ValueError("Model output extends beyond source video")
    except BaseException:
        if process.poll() is None:
            process.kill(); process.wait()
        raise
    finally:
        cap.release()


def state_examples(run: dict, split: str, clips: dict[str, dict], inputs: dict[str, dict], base: Path, target: Path) -> list[dict]:
    candidates: dict[int, list[dict]] = {}
    for clip_id, rec in run["records"].items():
        if rec["split"] != split or not rec["statistics"]["technical_output_valid"]:
            continue
        for s in rec["statistics"]["segments"]:
            candidates.setdefault(s["state"], []).append({"id": clip_id, **s})
    panels, selections = [], []
    for state, pieces in sorted(candidates.items()):
        chosen, used = [], set()
        for p in sorted(pieces, key=lambda p: (-p["seconds"], p["id"], p["start"])):
            if p["id"] not in used:
                chosen.append(p); used.add(p["id"])
            if len(chosen) == 3:
                break
        tiles = []
        for p in chosen:
            clip_id = p["id"]; source = inputs[clip_id]
            i = (p["start"] + p["end"] - 1) // 2
            cap = cv2.VideoCapture(str(base / clips[clip_id]["video"]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(source["frame_indices"][i]))
            ok, image = cap.read(); cap.release()
            if not ok:
                raise ValueError(f"Cannot decode {clip_id} frame {i}")
            image = draw(image, source, i, state, clip_id, split)
            tiles.append(cv2.resize(image, (384, 216)))
            selections.append({"state": state, "id": clip_id, "input_frame": i,
                               "source_frame": int(source["frame_indices"][i]), "seconds": int(source["frame_indices"][i]) / source["fps"]})
        while len(tiles) < 3:
            tiles.append(np.full((216, 384, 3), 245, np.uint8))
        panels.append(np.hstack(tiles))
    if panels and not cv2.imwrite(str(target), np.vstack(panels)):
        raise IOError(f"Cannot write {target}")
    return selections


def timeline(runs: list[dict], clip_ids: list[str], inputs: dict[str, dict], target: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [(run, clip_id) for clip_id in clip_ids for run in runs if clip_id in run["records"]
            and run["records"][clip_id]["statistics"]["technical_output_valid"]]
    if not rows:
        return
    figure, axes = plt.subplots(len(rows), 1, figsize=(13, max(3, len(rows) * 1.05)), squeeze=False)
    for axis, (run, clip_id) in zip(axes[:, 0], rows):
        rec = run["records"][clip_id]; fps = inputs[clip_id]["fps"]
        for s in rec["statistics"]["segments"]:
            axis.broken_barh([(s["start"] / fps, s["seconds"])], (0, 1), facecolors=[np.array(_color(s["state"]))[::-1] / 255])
            if s["seconds"] > .3:
                axis.text((s["start"] + s["end"]) / fps / 2, .5, str(s["state"]), ha="center", va="center", fontsize=8)
        axis.set(xlim=(0, rec["statistics"]["duration_seconds"]), ylim=(0, 1), yticks=[], ylabel=f"{clip_id}\n{run['name']}")
    axes[-1, 0].set_xlabel("Seconds within each independent clip; state IDs are not semantic action labels")
    figure.tight_layout(); figure.savefig(target, dpi=140); plt.close(figure)


def build(manifest_path: Path, run_dirs: list[Path], output: Path, media_clips: int = 3) -> Path:
    manifest = json.loads(manifest_path.read_text()); base = manifest_path.parent
    clips = manifest["clips"]
    if len({c["id"] for c in clips}) != len(clips):
        raise ValueError("Manifest clip IDs must be unique")
    output.mkdir(parents=True, exist_ok=True)
    inputs, input_failures = {}, []
    for clip in clips:
        if clip["split"] not in {"train", "validation"}:
            continue
        try:
            inputs[clip["id"]] = load_inputs(clip, base)
        except (KeyError, ValueError, OSError) as exc:
            input_failures.append({"id": clip["id"], "error": str(exc)})
    runs = [load_run(p, clips, inputs, output) for p in run_dirs]
    comparisons = [compare_runs(a, b, split) for a, b in itertools.combinations(runs, 2) for split in ("train", "validation")]
    valid_ids = [c["id"] for c in clips if c["split"] == "validation" and c["id"] in inputs]
    selected = [valid_ids[int(i)] for i in np.linspace(0, len(valid_ids) - 1, min(media_clips, len(valid_ids)))] if valid_ids and media_clips else []
    training_ids = [c["id"] for c in clips if c["split"] == "train" and c["id"] in inputs]
    timeline_ids = ([training_ids[len(training_ids) // 2]] if training_ids else []) + selected
    timeline_ids = [cid for cid in timeline_ids if any(cid in run["records"] and run["records"][cid]["statistics"]["technical_output_valid"] for run in runs)]
    if timeline_ids:
        timeline(runs, timeline_ids, inputs, output / "timeline.png")
    media_errors = []
    clips_by_id = {c["id"]: c for c in clips}
    for index, run in enumerate(runs):
        run["media"] = []; run["examples"] = {}
        if media_clips:
            for split in ("train", "validation"):
                name = f"run-{index + 1:02d}-{split}-examples.jpg"
                try:
                    samples = state_examples(run, split, clips_by_id, inputs, base, output / name)
                    if samples:
                        run["examples"][split] = {"image": name, "frames": samples}
                except (OSError, ValueError, RuntimeError) as exc:
                    media_errors.append({"run": run["name"], "split": split, "error": str(exc)})
            for clip_id in selected:
                rec = run["records"].get(clip_id)
                if not rec or not rec["statistics"]["technical_output_valid"]:
                    continue
                name = f"run-{index + 1:02d}-{clip_id}.mp4"
                try:
                    overlay(base / clips_by_id[clip_id]["video"], output / name, rec, inputs[clip_id])
                    run["media"].append({"id": clip_id, "video": name, "split": "validation"})
                except (OSError, ValueError, RuntimeError) as exc:
                    media_errors.append({"run": run["name"], "id": clip_id, "error": str(exc)})
    clean_runs = []
    for run in runs:
        clean = {k: v for k, v in run.items() if k != "records"}
        clean["records"] = {key: {k: v for k, v in rec.items() if k not in {"labels", "frame_indices"}} for key, rec in run["records"].items()}
        clean_runs.append(clean)
    report = {"scope": "same-camera-same-session-video-held-out", "manifest": relative(manifest_path, output),
              "split_counts": {s: sum(c["split"] == s for c in clips) for s in sorted({c["split"] for c in clips})},
              "input_statistics": {"readable_clips": len(inputs),
                                   "processed_fps": sorted({v["fps"] for v in inputs.values()}),
                                   "processed_frames": sum(len(v["coordinates"]) for v in inputs.values()),
                                   "zero_confidence_keypoint_fraction": (sum(int((v["confidences"] == 0).sum()) for v in inputs.values()) / sum(v["confidences"].size for v in inputs.values())) if inputs else None},
              "split_metadata": {k: v for k, v in manifest.items() if k != "clips"},
              "excluded_clips": [c for c in clips if c["split"] not in {"train", "validation"}],
              "input_failures": input_failures, "runs": clean_runs, "comparisons": comparisons,
              "media_selection": {"rule": "chronologically evenly spaced held-out videos; middle training clip for timeline", "validation_ids": selected},
              "media_errors": media_errors,
              "limitations": ["No human-labelled action ground truth; ARI/NMI are seed agreement, not accuracy.",
                              "The same camera and recording session do not establish cross-player/cross-camera generalization.",
                              "State IDs have no action names or coaching quality semantics; compare poses visually.",
                              "Per-video held-out inference requires model parameters and PCA to remain fitted to training only."]}
    (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "index.html").write_text(render_html(report, timeline_ids, output, base), encoding="utf-8")
    return output / "index.html"


def render_html(report: dict, timeline_ids: list[str], output: Path, base: Path) -> str:
    e = html.escape
    counts = " · ".join(f"{e(k)}：{v}段" for k, v in report["split_counts"].items())
    sections = []
    for run in report["runs"]:
        rows = []
        for split, a in run["aggregate"].items():
            median = f"{a['median_segment_seconds']:.3f}" if a["median_segment_seconds"] is not None else "—"
            tiny = f"{a['tiny_segment_fraction_under_0_1_seconds']:.1%}" if a["tiny_segment_fraction_under_0_1_seconds"] is not None else "—"
            rows.append(f'<tr><td>{e(split)}</td><td>{a["clips"]}</td><td>{a["frames"]}</td><td>{a["duration_seconds"]:.1f}s</td><td>{a["active_states"]}</td><td>{a["segment_count"]}</td><td>{median}s</td><td>{tiny}</td><td>{a["single_state_clip_count"]}</td></tr>')
        videos = "".join(f'<h3>留出视频 {e(m["id"])}</h3><video controls preload="metadata" src="{e(m["video"])}"></video>' for m in run["media"])
        examples = "".join(f'<details><summary>{e(split)}：每状态典型帧，优先取不同视频</summary><img src="{e(item["image"])}"></details>' for split, item in run["examples"].items())
        checkpoints = " · ".join(f'<a href="{e(p)}">checkpoint</a>' for p in run["checkpoints"])
        failures = "".join(f'<li>{e(f["id"])} ({e(f["split"])}): {e(f["error"])}</li>' for f in run["failures"])
        clip_rows = "".join(f'<tr><td>{e(cid)}</td><td>{e(r["split"])}</td><td>{"通过" if r["statistics"]["technical_output_valid"] else "失败"}</td><td>{r["statistics"]["active_states"]}</td><td>{r["statistics"]["segment_count"]}</td><td>{r["zero_confidence_fraction"]:.1%}</td><td><a href="{e(r["results"])}">NPZ</a></td></tr>' for cid, r in run["records"].items())
        sections.append(f'<section><h2>{e(run["name"])}</h2><p>运行状态：{e(str(run["summary"].get("status", "unknown")))}；持久化checkpoint：{"存在" if run["has_checkpoint"] else "缺失"}。<a href="{e(run["summary_link"])}">配置、计时与原始训练摘要</a> · {checkpoints}</p><p>推断前后参数SHA256一致性：{e(str(run["frozen_parameter_verification"]))}。</p><table><tr><th>集合</th><th>视频</th><th>帧</th><th>时长</th><th>状态</th><th>片段</th><th>中位时长</th><th>&lt;0.1s片段占比</th><th>单状态视频</th></tr>{"".join(rows)}</table><p>验证集中训练集未出现的状态：{e(str(run["aggregate"]["validation"]["states_absent_from_training"]))}。单状态片段可能是真正静止，也可能是塌缩，需结合视频判断；短片段统计包含每个文件首尾被截断的状态。</p><details><summary>对齐与heading配置</summary><pre>{e(json.dumps(run["summary"].get("alignment", {}), ensure_ascii=False, indent=2))}</pre></details><details><summary>训练/验证状态分布差异：JS divergence与total variation</summary><pre>{e(json.dumps(run["train_validation_occupancy_shift"], ensure_ascii=False, indent=2))}</pre></details>{examples}{videos}<details><summary>逐视频输出与数据质量</summary><table><tr><th>视频</th><th>集合</th><th>技术输出</th><th>状态</th><th>片段</th><th>零置信度点</th><th>文件</th></tr>{clip_rows}</table></details>{"<p>失败视频：</p><ul>" + failures + "</ul>" if failures else ""}</section>')
    comparison_rows = []
    details = []
    for c in report["comparisons"]:
        pooled = c["pooled"] or {}
        ari = f"{pooled['adjusted_rand_index']:.3f}" if pooled else "—"
        nmi = f"{pooled['normalized_mutual_information']:.3f}" if pooled else "—"
        median = f"{c['median_clip_ARI']:.3f}" if c["median_clip_ARI"] is not None else "—"
        comparison_rows.append(f'<tr><td>{e(c["left"])} / {e(c["right"])}</td><td>{e(c["split"])}</td><td>{c["clips"]}</td><td>{ari}</td><td>{nmi}</td><td>{median}</td><td>{c["degenerate_clip_count"]}</td></tr>')
        ordered = sorted(c["per_clip"], key=lambda p: p["adjusted_rand_index"])
        details.append(f'<details><summary>{e(c["split"])}逐视频一致性（低到高）</summary><pre>{e(json.dumps(ordered, ensure_ascii=False, indent=2))}</pre></details>')
    excluded = e(json.dumps(report["excluded_clips"], ensure_ascii=False, indent=2))
    errors = e(json.dumps({"input": report["input_failures"], "media": report["media_errors"]}, ensure_ascii=False, indent=2))
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>camera_05 · 定制MoSeq训练评估</title>'
            '<style>body{max-width:1160px;margin:34px auto;padding:0 22px;font:16px/1.7 system-ui;color:#1b2836}h1{font-size:29px}.notice{padding:16px;background:#fff3d8}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:8px;text-align:left;border-bottom:1px solid #ddd}video,img{width:100%;max-height:780px;object-fit:contain}section{border-top:1px solid #ddd;margin-top:28px;padding-top:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f9;padding:12px}summary{cursor:pointer}details{margin:16px 0}a{color:#215f98}</style>'
            '<h1>camera_05：定制 keypoint‑MoSeq 训练与留出视频评估</h1>'
            '<p class="notice">这是同一机位、同一次录像中的视频留出验证。没有人工动作真值，不能报告动作识别准确率；跨种子ARI/NMI衡量分段一致性。状态编号不是正手、反手等动作名称，也不代表动作质量。</p>'
            f'<p>{counts}；输入帧率 {e(str(report["input_statistics"]["processed_fps"]))} fps，处理后共 {report["input_statistics"]["processed_frames"]} 帧。<a href="{e(relative(base / "METHOD.html", output))}">处理与训练方法记录</a> · <a href="{e(report["manifest"])}">数据划分与输入清单</a> · <a href="validation.json">完整验证数据</a></p>'
            '<p>每段视频独立建模序列，分段统计不跨文件边界累计。训练集用于PCA和模型参数学习；留出视频只做固定参数推断。为保持可比，下面的train与validation输出均由训练后的同一固定参数流程重新推断，而非把拟合中间状态直接与留出状态比较。请在原始训练摘要中核对冻结参数校验记录。基准模型预先指定seed 0，其他种子仅检查稳定性，不按留出集表现挑选模型。</p>'
            + ('<h2>代表性视频的状态时间轴</h2><p>验证视频按时间顺序等间隔选择；另显示训练集中间视频。不同训练的相同编号/颜色不表示语义对应。</p><img src="timeline.png">' if timeline_ids else '')
            + '<h2>跨种子稳定性</h2><p>合并ARI/NMI检查跨视频的一致分组；逐视频中位数检查局部一致性。单状态结果即使ARI=1，也不能据此认定有效。</p><table><tr><th>模型</th><th>集合</th><th>共同视频</th><th>合并ARI</th><th>合并NMI</th><th>逐视频ARI中位数</th><th>退化对比数</th></tr>' + "".join(comparison_rows) + '</table>' + "".join(details)
            + "".join(sections)
            + f'<h2>排除与保留间隔</h2><p>以下文件不参与模型训练或验证；原因保留在清单中。</p><details><summary>查看排除清单</summary><pre>{excluded}</pre></details>'
            + f'<h2>处理失败记录</h2><pre>{errors}</pre>'
            + '<h2>判断输出是否有用</h2><ul><li>同状态应在不同视频中呈现相似姿态或运动阶段，并能在留出视频复现。</li><li>检查异常切换是否对应关节抖动、遮挡、错误主体或真正的动作改变。</li><li>训练参数固定后的留出分段仍不是语义正确性的证明，需要人工标注和更多人物、日期、机位验证。</li><li>本次是人体二维关键点聚类；不能据此测量球拍触球、握拍或真实三维关节角。</li></ul></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--media-clips", type=int, default=3)
    args = parser.parse_args()
    if args.media_clips < 0:
        parser.error("--media-clips must be nonnegative")
    print(build(args.manifest, args.runs, args.output, args.media_clips))


if __name__ == "__main__":
    main()
