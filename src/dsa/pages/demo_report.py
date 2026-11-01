"""An auditable analysis report for a processed clip, using Codex CLI only."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json

import cv2
import numpy as np

from dsa.codex import ask
from dsa.scene.results import load
from dsa.video import frame_at


SCHEMA = {"type": "object", "additionalProperties": False,
          "properties": {k: {"type": "array", "items": {"type": "string"}}
                         for k in ("observations", "limitations", "next_steps")},
          "required": ["observations", "limitations", "next_steps"]}


def build(spec: str):
    d = load(spec)
    n, fps, root = d["total"], d["fps"], d["src"]
    if n <= 0:
        raise ValueError("No analysed frames")
    z = np.load(root / "raw.npz")
    if len(z["n"]) != n or len(z["ball"]) != n:
        raise ValueError("Tracking and GPU output frame counts disagree")
    rows = []
    for hit in d["hits"]:
        f = hit["frame"]
        if not 0 <= f < n:
            raise ValueError(f"Hit frame outside clip: {f}")
        side = 0 if hit["side"] == "near" else 1
        rows.append({**hit, "clip_seconds": round(f / fps, 3),
                     "player_id": d["pl"]["near_far"][f][side]})
    counts = Counter(h["hand"] or h["shot"] for h in rows)
    summary = {"clip": spec, "frames": n, "fps": fps, "duration_seconds": n / fps,
               "ball_detected_frames": int((z["ball"][:, 0] > 0).sum()),
               "both_players_frames": sum(all(v is not None for v in p) for p in d["pl"]["near_far"]),
               "pose_crops": int(np.isfinite(z["kp"][:, 0, 0]).sum()),
               "detected_person_boxes": int(z["n"].sum()),
               "hit_count": len(rows), "predicted_stroke_counts": dict(counts),
               "gpu_stage_seconds": json.loads(str(z["seconds"])), "hits": rows,
               "llm_backend": "codex-cli", "identity": "Anonymous Player 1 / Player 2; names not verified"}
    summary["ball_detection_fraction"] = round(summary["ball_detected_frames"] / n, 4)
    summary["both_players_fraction"] = round(summary["both_players_frames"] / n, 4)
    summary["review_frames"] = [{"image": f"review_{i}.jpg", "frame": int(f),
                                "clip_seconds": round(int(f) / fps, 3)}
                               for i, f in enumerate(np.linspace(0, n - 1, 5, dtype=int))]
    (root / "analysis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    images = []
    for review in summary["review_frames"]:
        p = root / review["image"]
        cv2.imwrite(str(p), frame_at(str(root / "render.mp4"), review["frame"]))
        images.append(p)
    narrative, _ = ask(
        "请用中文为这段网球演示生成简洁分析。附件为带模型叠加的均匀抽帧，JSON为完整视频测量。"
        "区分画面观察和模型预测；不能从检测率推导准确率，不猜测球速、握拍、真实三维关节角度或伤病。"
        "球员姓名未经核实，用球员1/2。没有职业参考数据，不能宣称已和职业球员比较。"
        "没有足够样本的技术结论应明确保留。observations列出可核对的发现，limitations列出局限，"
        "next_steps列出改善拍摄/分析的具体下一步。数据：\n" + json.dumps(summary, ensure_ascii=False),
        images, SCHEMA, root / "report_cache")
    (root / "codex_analysis.json").write_text(json.dumps(narrative, ensure_ascii=False, indent=2))
    sections = "".join(f"<h2>{title}</h2><ul>" + "".join(f"<li>{html.escape(s)}</li>" for s in narrative[key])
                       + "</ul>" for key, title in [("observations", "观察结果"), ("limitations", "局限"),
                                                  ("next_steps", "下一步")])
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{n / fps:g}秒网球分析 · Codex</title>
<style>body{{max-width:1000px;margin:40px auto;padding:0 20px;font:17px/1.7 system-ui;color:#182235;background:#f6f8fb}}
video{{width:100%;background:#111;border-radius:12px}}h1,h2{{line-height:1.3}}li{{margin:10px 0}}a{{color:#1866b4}}</style>
<h1>{n / fps:g} 秒网球视频分析</h1><p>RTX 5090 D 视觉推理 · Codex 画面标定与结果解读</p>
<p>{n} 帧 · {fps:g} fps · {summary['hit_count']} 个预测击球事件。检测覆盖率不等于准确率。</p>
<h2>球员、骨架、球与击球标签</h2><video controls src="render.mp4"></video>
<h2>骨架视图</h2><video controls src="render_plain.mp4"></video>{sections}
<p><a href="analysis.json">完整测量 JSON</a> · <a href="codex_analysis.json">Codex 解读 JSON</a></p></html>"""
    (root / "index.html").write_text(page)
    return root / "index.html"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    print(build(ap.parse_args().video))
