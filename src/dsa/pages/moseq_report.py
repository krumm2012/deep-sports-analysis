"""Inspect short-video MoSeq outputs without treating training fit as accuracy.

    python -m dsa.pages.moseq_report --input moseq-input.npz --video source.mp4 \
        --runs moseq/seed0 moseq/seed1 --output moseq-review
"""
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import quote

import cv2
import numpy as np


BONES = [(1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (1, 7), (2, 8), (7, 8),
         (7, 9), (9, 11), (8, 10), (10, 12), (11, 13), (11, 14), (11, 15),
         (12, 16), (12, 17), (12, 18)]


def segments(labels: np.ndarray, fps: float) -> list[dict]:
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.isfinite(labels).all():
        raise ValueError("syllable must be a nonempty finite one-dimensional array")
    if not np.array_equal(labels, labels.astype(np.int64)) or (labels < 0).any():
        raise ValueError("syllable labels must be nonnegative integers")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    bounds = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    return [{"state": int(labels[a]), "start": int(a), "end": int(b),
             "frames": int(b - a), "seconds": float((b - a) / fps)}
            for a, b in zip(bounds[:-1], bounds[1:])]


def run_statistics(labels: np.ndarray, fps: float, expected_frames: int | None = None,
                   arrays: dict[str, np.ndarray] | None = None) -> dict:
    labels = np.asarray(labels)
    pieces = segments(labels, fps)
    if expected_frames is not None and len(labels) != expected_frames:
        raise ValueError(f"Label/input length mismatch: {len(labels)} != {expected_frames}")
    finite = {}
    for name, value in (arrays or {}).items():
        value = np.asarray(value)
        if not value.ndim or len(value) != len(labels):
            raise ValueError(f"{name}/label length mismatch")
        finite[name] = bool(np.isfinite(value).all())
    ids, counts = np.unique(labels, return_counts=True)
    durations = np.array([p["seconds"] for p in pieces])
    frequencies = [{"state": int(s), "frames": int(n), "fraction": float(n / len(labels)),
                    "segments": int(sum(p["state"] == s for p in pieces))} for s, n in zip(ids, counts)]
    maximum = float(counts.max() / len(labels))
    collapsed = len(ids) == 1
    return {"frames": int(len(labels)), "duration_seconds": float(len(labels) / fps),
            "active_states": int(len(ids)), "frequencies": frequencies,
            "segment_count": len(pieces), "transition_count": len(pieces) - 1,
            "transitions_per_second": float((len(pieces) - 1) / (len(labels) / fps)),
            "duration_quantiles_seconds": dict(zip(["p10", "median", "p90"],
                                                   np.quantile(durations, [.1, .5, .9]).tolist())),
            "tiny_segments_under_0_1_seconds": int((durations < .1 - 1e-9).sum()),
            "largest_state_fraction": maximum, "single_state_collapse": collapsed,
            "dominant_state_over_95_percent": maximum >= .95,
            "finite_outputs": finite, "technical_output_valid": all(finite.values()),
            "interpretation": "单状态，未形成可用动作分段" if collapsed else "存在分段；动作解释性仍需看原视频核验",
            "segments": pieces}


def compare_labels(a: np.ndarray, b: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or not len(a):
        raise ValueError("Comparisons require equal nonempty aligned label sequences")
    degenerate = len(np.unique(a)) == 1 or len(np.unique(b)) == 1
    return {"adjusted_rand_index": float(adjusted_rand_score(a, b)),
            "normalized_mutual_information": float(normalized_mutual_info_score(a, b)),
            "degenerate_comparison": degenerate,
            "interpretation": "至少一个结果只有单状态，不能凭高相似度判定有效" if degenerate
                              else "编号置换不影响此比较；数值仅反映同片段内分组一致性，不是动作准确率"}


def _color(state: int) -> tuple[int, int, int]:
    hue = int((int(state) * 37) % 180)
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, 190, 235]]]), cv2.COLOR_HSV2BGR)[0, 0])


def _draw(frame: np.ndarray, points: np.ndarray, confidence: np.ndarray, state: int, time: float) -> np.ndarray:
    image = frame.copy()
    color = _color(state)
    seen = np.isfinite(points).all(1) & (confidence >= .1)
    for a, b in BONES:
        if a < len(points) and b < len(points) and seen[a] and seen[b]:
            cv2.line(image, tuple(np.rint(points[a]).astype(int)), tuple(np.rint(points[b]).astype(int)), color, 3, cv2.LINE_AA)
    for point in points[seen]:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA)
    y = image.shape[0] - 48
    image[y:] = (image[y:] * .25).astype(np.uint8)
    cv2.putText(image, f"State {state} | {time:.2f}s | in-sample fit", (15, y + 31),
                cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
    return image


def render_overlay(video: Path, target: Path, labels: np.ndarray, inputs: dict) -> None:
    cap = cv2.VideoCapture(str(video))
    width, height = (int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        raise ValueError(f"Unreadable video: {video}")
    fps = float(inputs["fps"])
    lookup = {int(frame): i for i, frame in enumerate(inputs["frame_indices"])}
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no in lookup:
                i = lookup[frame_no]
                frame = _draw(frame, inputs["image_coordinates"][i], inputs["confidences"][i], int(labels[i]), frame_no / fps)
            process.stdin.write(frame.tobytes())
            frame_no += 1
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        code = process.wait()
        if code:
            raise RuntimeError(f"Video encoding failed: {error}")
        if max(lookup, default=-1) >= frame_no:
            raise ValueError("Training frame index extends beyond source video")
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        cap.release()


def representative_frames(video: Path, target: Path, labels: np.ndarray, inputs: dict) -> None:
    pieces = segments(labels, float(inputs["fps"]))
    cap = cv2.VideoCapture(str(video))
    rows = []
    for state in np.unique(labels):
        candidates = sorted([p for p in pieces if p["state"] == state], key=lambda p: -p["frames"])
        chosen = [(p["start"] + p["end"] - 1) // 2 for p in candidates[:3]]
        if len(chosen) < 3:
            longest = candidates[0]
            chosen += [int(v) for v in np.linspace(longest["start"], longest["end"] - 1, 3) if int(v) not in chosen]
        tiles = []
        for i in chosen[:3]:
            frame_no = int(inputs["frame_indices"][i])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Cannot read representative source frame {frame_no}")
            image = _draw(image, inputs["image_coordinates"][i], inputs["confidences"][i], int(state), frame_no / float(inputs["fps"]))
            tiles.append(cv2.resize(image, (384, 216)))
        while len(tiles) < 3:
            tiles.append(np.full((216, 384, 3), 245, np.uint8))
        rows.append(np.hstack(tiles))
    cap.release()
    if not cv2.imwrite(str(target), np.vstack(rows)):
        raise IOError(f"Cannot write {target}")


def timeline(records: list[dict], target: Path, fps: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(records), 1, figsize=(12, max(2.2, 1.5 * len(records))), squeeze=False)
    for axis, record in zip(axes[:, 0], records):
        for piece in record["statistics"]["segments"]:
            color = np.array(_color(piece["state"]))[::-1] / 255
            axis.broken_barh([(piece["start"] / fps, piece["seconds"])], (0, 1), facecolors=[color])
            if piece["seconds"] > .2:
                axis.text((piece["start"] + piece["end"]) / fps / 2, .5, str(piece["state"]), ha="center", va="center", fontsize=8)
        axis.set(xlim=(0, record["statistics"]["duration_seconds"]), ylim=(0, 1), yticks=[], ylabel=record["name"])
    axes[-1, 0].set_xlabel("Seconds in training sequence (state IDs have no action semantics)")
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)


def _link(path: Path, output: Path) -> str:
    return quote(os.path.relpath(path.resolve(), output.resolve()), safe="/")


def build(input_path: Path, video: Path, runs: list[Path], output: Path) -> Path:
    data = np.load(input_path, allow_pickle=False)
    required = ["coordinates", "confidences", "image_coordinates", "bodyparts", "fps", "frame_indices"]
    inputs = {k: data[k] for k in required}
    fps, n = float(inputs["fps"]), len(inputs["coordinates"])
    if not n or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Input must contain frames and a positive frame rate")
    if inputs["coordinates"].shape != inputs["image_coordinates"].shape or inputs["coordinates"].shape != (n, 19, 2):
        raise ValueError("Expected normalized and image coordinates shaped (T,19,2)")
    if inputs["confidences"].shape != (n, 19) or inputs["frame_indices"].shape != (n,):
        raise ValueError("Input confidence/frame alignment mismatch")
    if not np.isfinite(inputs["confidences"]).all() or (inputs["confidences"] < 0).any() or (inputs["confidences"] > 1).any():
        raise ValueError("Input confidences must be finite values in [0,1]")
    if not np.array_equal(inputs["frame_indices"], inputs["frame_indices"].astype(np.int64)) or (inputs["frame_indices"] < 0).any():
        raise ValueError("Source frame indices must be nonnegative integers")
    if np.any(np.diff(inputs["frame_indices"]) <= 0):
        raise ValueError("Source frame indices must be unique and increasing")
    output.mkdir(parents=True, exist_ok=True)
    original_ids = data["source_original_frame_indices"] if "source_original_frame_indices" in data else None
    source_info = {"frames_used": n, "fps": fps, "duration_seconds": n / fps,
                   "unique_original_frames": int(len(np.unique(original_ids))) if original_ids is not None else None,
                   "confidence_zero_fraction": float((inputs["confidences"] == 0).mean()),
                   "metadata": json.loads(str(data["metadata"])) if "metadata" in data else {}}
    records, failures = [], []
    for index, folder in enumerate(runs):
        try:
            result = np.load(folder / "results.npz", allow_pickle=False)
            labels = result["syllable"]
            arrays = {k: result[k] for k in ("latent_state", "centroid", "heading")}
            stats = run_statistics(labels, fps, n, arrays)
            if "frame_indices" in result and not np.array_equal(result["frame_indices"], inputs["frame_indices"]):
                raise ValueError("Training/source frame indices differ")
            summary = json.loads((folder / "summary.json").read_text()) if (folder / "summary.json").exists() else {}
            if str(summary.get("status", "")).lower() in {"failed", "error", "incomplete", "aborted", "stopped"}:
                stats["technical_output_valid"] = False
                stats["interpretation"] = "训练未成功完成；不能将部分输出判定为有效模型"
            prefix = f"run-{index + 1:02d}"
            record = {"name": folder.name, "directory": str(folder.resolve()), "statistics": stats,
                      "summary": summary, "video": prefix + ".mp4", "examples": prefix + "-examples.jpg",
                      "results": _link(folder / "results.npz", output),
                      "checkpoints": [_link(p, output) for p in sorted(folder.rglob("*checkpoint*.h5"))[:5]]}
            if stats["technical_output_valid"]:
                render_overlay(video, output / record["video"], labels, inputs)
                representative_frames(video, output / record["examples"], labels, inputs)
            else:
                record["video"] = record["examples"] = None
            records.append({**record, "labels": labels})
        except (KeyError, ValueError, OSError) as error:
            failures.append({"name": folder.name, "error": str(error)})
    comparisons = []
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            comparisons.append({"left": left["name"], "right": right["name"],
                                **compare_labels(left["labels"], right["labels"])})
    if records:
        timeline(records, output / "timeline.png", fps)
    report = {"validation_scope": "same-short-clip-in-sample-only", "source": source_info,
              "runs": [{k: v for k, v in r.items() if k != "labels"} for r in records],
              "failed_runs": failures, "comparisons": comparisons,
              "not_a_generalization_or_accuracy_measurement": True}
    (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    e = html.escape
    sections = []
    for record in records:
        stats = record["statistics"]
        rows = "".join(f'<tr><td>{v["state"]}</td><td>{v["frames"]}</td><td>{v["fraction"]:.1%}</td><td>{v["segments"]}</td></tr>' for v in stats["frequencies"])
        visuals = (f'<video controls preload="metadata" src="{e(record["video"])}"></video>'
                   f'<details><summary>每状态典型帧：仅显示编号，不赋予动作名称</summary><img src="{e(record["examples"])}"></details>') if record["video"] else '<p>输出含非有限值，未生成叠加视频。</p>'
        checkpoint_links = " · ".join(f'<a href="{e(p)}">checkpoint</a>' for p in record["checkpoints"]) or "未找到持久化 checkpoint"
        sections.append(f'<section><h2>{e(record["name"])}</h2><p><b>{e(stats["interpretation"])}</b></p>'
                        f'<p>技术输出：{"通过" if stats["technical_output_valid"] else "未通过"}；有效状态 {stats["active_states"]}；'
                        f'片段 {stats["segment_count"]}；中位持续 {stats["duration_quantiles_seconds"]["median"]:.3f}s；'
                        f'持续时长P10–P90 {stats["duration_quantiles_seconds"]["p10"]:.3f}–{stats["duration_quantiles_seconds"]["p90"]:.3f}s；'
                        f'切换率 {stats["transitions_per_second"]:.2f}/s；短于0.1s的片段 {stats["tiny_segments_under_0_1_seconds"]}。</p>'
                        f'{visuals}<table><tr><th>状态</th><th>帧数</th><th>占比</th><th>片段数</th></tr>{rows}</table>'
                        f'<p><a href="{e(record["results"])}">输出 NPZ</a> · {checkpoint_links}</p>'
                        f'<details><summary>训练配置、计时与原始摘要</summary><pre>{e(json.dumps(record["summary"], ensure_ascii=False, indent=2))}</pre></details></section>')
    comparison_rows = "".join(f'<tr><td>{e(c["left"])} / {e(c["right"])}</td><td>{c["adjusted_rand_index"]:.3f}</td>'
                              f'<td>{c["normalized_mutual_information"]:.3f}</td><td>{e(c["interpretation"])}</td></tr>' for c in comparisons)
    failed = "".join(f'<p>{e(f["name"])}：{e(f["error"])}</p>' for f in failures)
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>新场景网球视频 · MoSeq 训练验证</title><style>body{max-width:1100px;margin:36px auto;padding:0 22px;font:16px/1.7 system-ui;color:#19212b}h1{font-size:28px}.notice{background:#fff4db;padding:16px 20px}section{border-top:1px solid #ddd;margin-top:28px;padding-top:15px}video{width:100%;max-height:650px;background:#111}img{max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f5f5;padding:12px}a{color:#245b91}summary{cursor:pointer}details{margin:14px 0}</style>'
            '<h1>新场景网球视频：keypoint‑MoSeq 训练验证</h1>'
            '<p class="notice">本次使用同一段约8秒录像训练并检查输出，属于片段内验证。重采样重复帧不是新增独立样本。'
            '没有人工动作真值，不能给出动作识别准确率；技术上能生成结果，不等于具备跨视频泛化能力。</p>'
            '<p>先看视频中的状态切换是否对应可见的动作变化，再比较不同种子/参数。状态编号只是当前模型的类别编号，颜色与编号不代表动作好坏。</p>'
            f'<p>训练帧 {n}；30Hz重采样后时长 {n / fps:.3f}s；不同原始帧 {source_info["unique_original_frames"]}；'
            f'零置信度点占比 {source_info["confidence_zero_fraction"]:.1%}。'
            f'<a href="{e(_link(input_path, output))}">输入 NPZ</a> · <a href="{e(_link(video, output))}">标准化源视频</a> · <a href="validation.json">完整验证数据</a></p>'
            + ('<h2>分段时间线</h2><img src="timeline.png" alt="各训练结果的状态时间线">' if records else '')
            + "".join(sections)
            + ('<h2>同片段结果一致性</h2><p>ARI/NMI不受类别编号置换影响。高分不证明动作语义正确；单状态结果即使相似度为1也可能是塌缩。</p><table><tr><th>对比</th><th>ARI</th><th>NMI</th><th>解释</th></tr>' + comparison_rows + '</table>' if comparisons else '')
            + ('<h2>未通过检查的运行</h2>' + failed if failures else '')
            + '<h2>如何判断有用</h2><ul><li>同状态的多个片段应具有可识别且相似的动作，而非仅按时间先后划分。</li>'
            '<li>检查模型是否只学到机位、抖动或插值；当前短片不能证明它学到了通用网球动作。</li>'
            '<li>检查单状态塌缩、过多短片段以及不同种子间的大幅变化。若出现这些现象，应保留失败结果。</li>'
            '<li>后续需要更长、多回合、多机位数据和独立验证集；同视频内分段不应解释为教练评分。</li></ul></html>')
    (output / "index.html").write_text(page)
    return output / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.input, args.video, args.runs, args.output))


if __name__ == "__main__":
    main()
