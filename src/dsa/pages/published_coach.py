"""Replay published tennis measurements through Codex, without claiming a video re-analysis.

    python -m dsa.pages.published_coach --model MODEL

The public article embeds pro percentiles and guest medians, but does not publish
guest sample counts or their percentiles. Keep those missing fields explicit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import statistics
from urllib.parse import quote, urlparse

from jsonschema import Draft202012Validator


ARTICLE = "https://offpolicyinc.com/work/claude-tennis-coach/"
KINDS = ("serve", "forehand", "backhand")
LABELS = {"serve": "发球", "forehand": "正手", "backhand": "反手"}
# Mirrors the phone camera exclusions in profiles_report.SENS. Importing that
# report also imports the entire measurement stack; this replay needs no GPU.
HIGH_CAMERA = {"hip_coil", "separation", "contact_height", "finish_height", "reach", "toss_arm",
               "elbow_angle", "toss_straight", "knee_bend", "shoulder_tilt", "trophy_tilt", "wrist_speed"}
LOW_CAMERA = {"stance_m", "stance_side_m", "stance_fwd_m", "stance_angle", "swing_time", "trophy_time", "into_court"}


class _PayloadParser(HTMLParser):
    def __init__(self, script_id: str):
        super().__init__(convert_charrefs=False)
        self.script_id = script_id
        self.active = False
        self.chunks: list[str] = []
        self.matches = 0

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("id") == self.script_id:
            self.active = True
            self.matches += 1

    def handle_endtag(self, tag):
        if tag == "script":
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.chunks.append(data)


def read_payload(snapshot: Path, script_id: str = "ex-data") -> dict:
    parser = _PayloadParser(script_id)
    parser.feed(snapshot.read_text())
    if parser.matches != 1:
        raise ValueError(f"Expected one #{script_id} data block, found {parser.matches}")
    data = json.loads("".join(parser.chunks))
    if not isinstance(data, dict) or not isinstance(data.get("measures"), list):
        raise ValueError("Published data has no measures list")
    return data


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def player_input(data: dict, player: str, snapshot: Path, script_id: str = "ex-data") -> dict:
    if player not in data.get("ours", []):
        raise ValueError(f"{player} is not a guest in #{script_id}")
    guests = set(data["ours"])
    metrics, excluded, ids = [], [], set()
    for row in data["measures"]:
        kind, name = row["k"], row["n"]
        if kind not in KINDS:
            continue
        metric_id = f"{kind}.{name}"
        if metric_id in ids:
            raise ValueError(f"Duplicate published metric: {metric_id}")
        ids.add(metric_id)
        med = row.get("meds", {}).get(player)
        z = row.get("z", {}).get(player)
        if name in HIGH_CAMERA or not (_finite(med) and _finite(z)):
            excluded.append({"id": metric_id, "reason": "high camera sensitivity" if name in HIGH_CAMERA
                             else "guest measurement missing"})
            continue
        pros = {p: v for p, v in row.get("meds", {}).items() if p not in guests and _finite(v)}
        if len(pros) < 2:
            raise ValueError(f"Insufficient pro comparison for {metric_id}")
        box = row.get("box", {}).get(player)
        iqr = [box[1], box[3]] if isinstance(box, list) and len(box) == 5 and all(_finite(v) for v in box) else None
        metrics.append({"id": metric_id, "kind": kind, "name": name,
                        "label": row.get("short", name), "description": row.get("say", ""),
                        "unit": row.get("unit", "").strip(), "player_median": med, "player_iqr": iqr,
                        "pro_median": statistics.median(pros.values()), "pro_min": min(pros.values()),
                        "pro_max": max(pros.values()), "pro_player_count": len(pros),
                        "pro_player_medians": pros, "z": z,
                        "camera_sensitivity": "low" if name in LOW_CAMERA else "medium"})
    if any(not any(m["kind"] == k for m in metrics) for k in KINDS):
        raise ValueError(f"{player} lacks usable measurements for one or more stroke kinds")
    clips = {}
    for kind in KINDS:
        c = data.get("clips", {}).get(kind, {})
        examples = c.get("players", {}).get(player, [])
        if not examples or not c.get("avg"):
            raise ValueError(f"Missing published {kind} videos for {player}")
        clips[kind] = {"player": examples[0], "reference": c["avg"], "all_player_examples": examples}
    return {"source": {"url": ARTICLE, "snapshot": str(snapshot.resolve()),
                       "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                       "data_script_id": script_id, "mode": "published-data-replay",
                       "not_recomputed_from_source_video": True},
            "player": player, "sample_counts": dict.fromkeys(KINDS), "sample_count_status": "unknown",
            "metrics": metrics, "excluded_metrics": excluded, "clips": clips,
            "limitations": ["使用作者已发布的汇总测量，未从源视频重新计算。",
                            "公开数据没有该球员逐类有效击球数；不能用五个展示片段替代样本量。",
                            "公开数据没有该球员的四分位范围；缺失值保留为 null。",
                            "二维投影和手机机位可能影响指标；差异不能证明缺陷或训练建议的因果效果。",
                            "职业范围是各职业球员中位数的范围；z 是标准化差异，不是技术得分或显著性概率。"]}


def output_schema(inputs: dict) -> dict:
    ids = [m["id"] for m in inputs["metrics"]]
    evidence = {"type": "array", "items": {"type": "string", "enum": ids}, "minItems": 1, "maxItems": 3}
    point = {"type": "object", "additionalProperties": False, "required": ["text", "evidence_ids"],
             "properties": {"text": {"type": "string"}, "evidence_ids": evidence}}
    stroke = {"type": "object", "additionalProperties": False,
              "required": ["kind", "headline", "lead", "points", "fix", "evidence_ids"],
              "properties": {"kind": {"type": "string", "enum": list(KINDS)},
                             "headline": {"type": "string"}, "lead": {"type": "string"},
                             "points": {"type": "array", "items": point, "minItems": 1, "maxItems": 2},
                             "fix": {"type": "string"}, "evidence_ids": evidence}}
    return {"type": "object", "additionalProperties": False,
            "required": ["strokes", "priority", "limitations"],
            "properties": {"strokes": {"type": "array", "items": stroke, "minItems": 3, "maxItems": 3},
                           "priority": {"type": "object", "additionalProperties": False,
                                        "required": ["kind", "text", "evidence_ids"],
                                        "properties": {"kind": {"type": "string", "enum": list(KINDS)},
                                                       "text": {"type": "string"}, "evidence_ids": evidence}},
                           "limitations": {"type": "array", "items": {"type": "string"}, "minItems": 1}}}


def validate_answer(answer: dict, inputs: dict, schema: dict | None = None) -> None:
    Draft202012Validator(schema or output_schema(inputs)).validate(answer)
    if sorted(s["kind"] for s in answer["strokes"]) != sorted(KINDS):
        raise ValueError("Report must contain each stroke kind exactly once")
    by_id = {m["id"]: m for m in inputs["metrics"]}
    for section in [*answer["strokes"], answer["priority"]]:
        for refs in [section["evidence_ids"], *(p["evidence_ids"] for p in section.get("points", []))]:
            if any(by_id[r]["kind"] != section["kind"] for r in refs):
                raise ValueError(f"Evidence must belong to {section['kind']}")


def make_prompt(inputs: dict) -> str:
    # Send only measurement context. The historical Claude prose and video URLs
    # are deliberately not included: no claims of watching unavailable video.
    data = {k: v for k, v in inputs.items() if k != "clips"}
    return ("请用中文为这位业余网球球员生成简洁、审慎的教练建议。你只收到作者公开的汇总测量文本，"
            "没有观看比赛视频；这次是 published-data replay，不是从录像重新测量。不要模仿或引用历史 Claude 建议。\n"
            "发球、正手、反手各输出一项最值得进一步检查的动作差异；已经接近职业参考时可以明确说明。"
            "每项用短标题、一两句解读、1–2条有依据的观察及一个低强度可实践的练习。"
            "每条观察和每类建议必须引用输入中同类动作的真实 metric id，不能创造指标或数值。"
            "只有职业球员中位数之间的范围，不能称为所有职业击球的范围。z不是好坏评分或显著性概率。"
            "职业差异不能证明动作错误，也不能证明改动一定有效。\n"
            "缺失数据都是未知：样本数 unknown，player_iqr=null 时不能猜测样本量、稳定性或显著性。"
            "手机机位敏感性 medium 的指标要保留视角影响；肩髋转角是二维投影代理，不能当成真实三维角度。"
            "不推断握拍、球速、伤病、战术、拍间步法或未测量的髋部动作，不作医疗诊断。"
            "建议使用待验证措辞。最后给出一项练习优先级，并列出样本数未知、二维/机位和汇总数据复用的局限。"
            "所有答案仅按指定JSON结构输出。数据：\n" + json.dumps(data, ensure_ascii=False, indent=2))


def _media_url(original: str, base: str) -> str:
    name = Path(urlparse(original).path).name
    if not name.endswith(".mp4") or not name:
        raise ValueError(f"Invalid published clip: {original}")
    if not base or base.lower().startswith(("javascript:", "data:")):
        raise ValueError("Unsafe media base")
    return base.rstrip("/") + "/" + quote(name)


def render_report(records: list[dict], media_base: str) -> str:
    e = html.escape
    models = ", ".join(sorted({r.get("model", "unspecified") for r in records}))
    blocks = []
    for index, record in enumerate(records):
        inputs, answer = record["input"], record["answer"]
        player = inputs["player"]
        by_id = {m["id"]: m for m in inputs["metrics"]}
        slug = player.lower().replace(" ", "-")
        def references(refs):
            return "、".join(f'<a href="#{e(slug)}-{e(r)}">{e(r)}</a>' for r in refs)
        sections = []
        for stroke in sorted(answer["strokes"], key=lambda s: KINDS.index(s["kind"])):
            kind = stroke["kind"]
            videos = []
            for role, caption in (("player", player), ("reference", "职业典型片段")):
                original, label = inputs["clips"][kind][role]
                videos.append(f'<figure><video controls muted playsinline preload="metadata" src="{e(_media_url(original, media_base))}"></video>'
                              f'<figcaption>{e(caption)} · {e(label)}</figcaption></figure>')
            points = "".join(f'<li>{e(p["text"])} <small>依据：{references(p["evidence_ids"])}</small></li>'
                             for p in stroke["points"])
            sections.append(f'<section class="stroke"><h3>{LABELS[kind]}</h3><div><h4>{e(stroke["headline"])}</h4>'
                            f'<p>{e(stroke["lead"])}</p><div class="videos">{"".join(videos)}</div><ul>{points}</ul>'
                            f'<p class="fix"><b>练习</b> {e(stroke["fix"])}</p>'
                            f'<small>本项依据：{references(stroke["evidence_ids"])}</small></div></section>')
        rows = []
        for metric_id, m in by_id.items():
            number = lambda v: f"{v:.4g}"
            rows.append(f'<tr id="{e(slug)}-{e(metric_id)}"><td>{e(metric_id)}</td><td>{e(m["label"])}</td>'
                        f'<td>{number(m["player_median"])} {e(m["unit"])}</td>'
                        f'<td>{number(m["pro_median"])}</td><td>{number(m["pro_min"])}–{number(m["pro_max"])}</td>'
                        f'<td>{m["z"]:+.2f}</td><td>{e(m["camera_sensitivity"])}</td></tr>')
        limitations = "".join(f"<li>{e(t)}</li>" for t in dict.fromkeys(inputs["limitations"] + answer["limitations"]))
        priority = answer["priority"]
        blocks.append(f'<article id="player-{e(slug)}" {"hidden" if index else ""}>'
                      f'{"".join(sections)}<p class="priority"><b>优先练习</b> {e(priority["text"])} '
                      f'<small>依据：{references(priority["evidence_ids"])}</small></p>'
                      f'<details><summary>全部测量、输入与局限</summary><div class="table-scroll"><table><thead>'
                      '<tr><th>指标 ID</th><th>含义</th><th>球员中位数</th><th>职业中位数</th><th>职业中位数范围</th><th>z</th><th>机位敏感性</th></tr>'
                      f'</thead><tbody>{"".join(rows)}</tbody></table></div><ul>{limitations}</ul>'
                      f'<p>有效击球数：未知；球员四分位范围：未公开。职业参考人数：{max(m["pro_player_count"] for m in by_id.values())}。</p>'
                      f'<p><a href="{e(slug)}/input.json">输入数据</a> · <a href="{e(slug)}/prompt.txt">完整业务提示词</a> · '
                      f'<a href="{e(slug)}/schema.json">输出约束</a> · <a href="{e(slug)}/answer.json">模型原始结果</a> · '
                      f'<a href="{e(slug)}/provenance.json">来源与运行记录</a></p></details></article>')
    buttons = "".join(f'<button data-player="{e(r["input"]["player"].lower().replace(" ", "-"))}" '
                      f'aria-pressed="{str(i == 0).lower()}">{e(r["input"]["player"])}</button>' for i, r in enumerate(records))
    return ("<!doctype html><html lang=\"zh-CN\"><meta charset=\"utf-8\">"
            '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Codex 网球教练 · 公开数据复刻</title>'
            '<style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#191919;background:#fff}'
            'h1{font-size:27px}h4{font-size:20px;margin:0 0 12px}header{border-top:3px solid #111;padding-top:16px}'
            '.notice{background:#f4f3ef;padding:15px 18px}.tabs{display:flex;gap:0;justify-content:flex-end}'
            'button{padding:10px 18px;border:1px solid #ddd;background:white;font:inherit;cursor:pointer}'
            'button[aria-pressed=true]{background:#111;color:#fff}.stroke{display:grid;grid-template-columns:130px 1fr;gap:20px;padding:28px 0;border-bottom:1px solid #ddd}'
            '.stroke h3{margin:0}.videos{display:flex;gap:12px}.videos figure{margin:0;width:min(45%,310px)}video{width:100%;background:#eee}'
            'figcaption,small{color:#666;font-size:13px}small{display:block}li{margin:9px 0}.fix,.priority{background:#f1f0ec;padding:14px 18px}'
            'a{color:#24578b}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}'
            '.table-scroll{overflow:auto}tr:target{background:#fff2ca}details{margin:24px 0}summary{cursor:pointer}'
            '@media(max-width:650px){.stroke{grid-template-columns:1fr;gap:8px}.videos figure{width:50%}body{padding:0 15px;margin-top:20px}}</style>'
            '<header><h1>Codex 的网球教练建议</h1><p class="notice">公开数据复刻：使用作者发布的汇总测量重新生成中文建议。'
            '未从原始录像重新计算，视频仅作示例；建议仍需在练习和教练指导中验证。</p>'
            f'<p>模型：{e(models)} · Codex CLI · 新生成建议</p>'
            f'<p><a href="{ARTICLE}#figure-7">原文章与历史 Claude 建议</a> · '
            '<a href="reference-review.html">文章与全部 9 条引用核查</a> · '
            '<a href="../published-demo/offline.html#figure-4">本地视频与原热力图</a> · '
            '<a href="deployment-status.html">5090 下载与处理状态</a> · '
            '<a href="manifest.json">本次运行清单</a></p>'
            f'<nav class="tabs" aria-label="选择球员">{buttons}</nav></header>{"".join(blocks)}'
            '<script>document.querySelectorAll("[data-player]").forEach(b=>b.addEventListener("click",()=>{'
            'document.querySelectorAll("[data-player]").forEach(x=>x.setAttribute("aria-pressed",String(x===b)));'
            'document.querySelectorAll("article").forEach(x=>x.hidden=x.id!=="player-"+b.dataset.player);'
            'document.querySelectorAll("video").forEach(v=>v.pause());}));</script></html>')


def build(snapshot: Path, output: Path, model: str, players=("Peter", "Richard"),
          media_base: str = "../published-media/", script_id: str = "ex-data", prepare_only: bool = False) -> Path:
    if not model.strip():
        raise ValueError("An explicit Codex model is required (--model or DSA_CODEX_MODEL)")
    from dsa.codex import ask

    data = read_payload(snapshot, script_id)
    output.mkdir(parents=True, exist_ok=True)
    records, runs = [], []
    for player in players:
        inputs = player_input(data, player, snapshot, script_id)
        prompt, schema = make_prompt(inputs), output_schema(inputs)
        folder = output / player.lower().replace(" ", "-")
        folder.mkdir(parents=True, exist_ok=True)
        for name, obj in (("input.json", inputs), ("schema.json", schema)):
            (folder / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        (folder / "prompt.txt").write_text(prompt)
        provenance = {"mode": "published-data-replay", "backend": "codex-cli", "model": model, "effort": "high",
                      "images_sent": 0, "videos_sent": 0, "source": inputs["source"],
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                      "prepared_at_utc": datetime.now(timezone.utc).isoformat(), "status": "prepared"}
        (folder / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
        if not prepare_only:
            answer, seconds = ask(prompt, [], schema, folder / "cache", model=model, effort="high")
            validate_answer(answer, inputs, schema)
            (folder / "answer.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2))
            provenance.update(status="complete", elapsed_seconds=seconds, generated_at_utc=datetime.now(timezone.utc).isoformat())
            (folder / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
            records.append({"input": inputs, "answer": answer, "model": model})
        runs.append({"player": player, "directory": folder.name, "status": provenance["status"],
                     "metric_count": len(inputs["metrics"]), "model": model})
    (output / "manifest.json").write_text(json.dumps({"mode": "published-data-replay", "players": runs}, ensure_ascii=False, indent=2))
    if not prepare_only:
        (output / "index.html").write_text(render_report(records, media_base))
    return output / ("manifest.json" if prepare_only else "index.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, default=Path("data/published-demo/original.html"))
    parser.add_argument("--output", type=Path, default=Path("data/reproduction"))
    parser.add_argument("--model", default=os.environ.get("DSA_CODEX_MODEL", ""))
    parser.add_argument("--players", nargs="+", default=["Peter", "Richard"])
    parser.add_argument("--media-base", default="../published-media/")
    parser.add_argument("--script-id", default="ex-data")
    parser.add_argument("--prepare-only", action="store_true", help="Export inputs without calling Codex")
    args = parser.parse_args()
    print(build(args.snapshot, args.output, args.model, args.players, args.media_base, args.script_id, args.prepare_only))


if __name__ == "__main__":
    main()
