"""Public measurements retain missing data and constrain coaching evidence."""
import copy
import json

import pytest
from jsonschema import ValidationError

from dsa.pages.published_coach import (KINDS, make_prompt, output_schema, player_input,
                                      read_payload, render_report, validate_answer)


@pytest.fixture
def published(tmp_path):
    rows = []
    for kind in KINDS:
        rows.append({"k": kind, "n": "swing_time", "short": "swing time", "say": "Time before contact.",
                     "unit": " s", "meds": {"Pro A": .1, "Pro B": .2, "Peter": .3, "Richard": .25},
                     "z": {"Peter": 1.2, "Richard": .8},
                     "box": {"Pro A": [.05, .08, .1, .12, .15], "Pro B": [.1, .15, .2, .25, .3]}})
    rows.append({"k": "backhand", "n": "hip_coil", "meds": {"Pro A": 80, "Pro B": 120, "Peter": 40}, "z": {"Peter": -2}})
    rows.append({"k": "serve", "n": "hips_over", "meds": {"Pro A": .1, "Pro B": .2}, "z": {}})
    clips = {kind: {"avg": [f"https://media.offpolicyinc.com/{kind}_pro.mp4", "Pro A"],
                    "players": {p: [[f"https://media.offpolicyinc.com/{kind}_{p}.mp4", p]] for p in ["Peter", "Richard"]}}
             for kind in KINDS}
    data = {"ours": ["Peter", "Richard"], "measures": rows, "clips": clips}
    path = tmp_path / "article.html"
    path.write_text('<script id="unrelated" type="application/json">{"measures":[]}</script>'
                    '<script type="application/json" id="ex-data">' + json.dumps(data) + '</script>'
                    '<p>Old advice says 43 degrees, do not use as data.</p>')
    return path, data


def answer_for(inputs):
    return {"strokes": [{"kind": k, "headline": "待检查的差异", "lead": "仅为汇总数据观察", "fix": "轻松练习",
                         "evidence_ids": [f"{k}.swing_time"],
                         "points": [{"text": "挥拍时间有差异", "evidence_ids": [f"{k}.swing_time"]}]} for k in KINDS],
            "priority": {"kind": "backhand", "text": "先复核反手", "evidence_ids": ["backhand.swing_time"]},
            "limitations": ["样本数未知"]}


def test_embedded_public_data_keeps_missing_counts_and_excludes_old_or_unreliable_claims(published):
    path, _ = published
    inputs = player_input(read_payload(path), "Peter", path)
    assert len(inputs["metrics"]) == 3
    assert all(m["player_iqr"] is None for m in inputs["metrics"])
    assert inputs["sample_counts"] == dict.fromkeys(KINDS)
    assert inputs["metrics"][0]["pro_median"] == pytest.approx(.15)
    assert inputs["metrics"][0]["pro_max"] == .2  # Guest medians are not pros.
    assert {r["id"] for r in inputs["excluded_metrics"]} == {"backhand.hip_coil", "serve.hips_over"}
    assert "43 degrees" not in make_prompt(inputs)
    assert "https://media.offpolicyinc.com/" not in make_prompt(inputs)


def test_noisy_or_duplicated_input_is_rejected(published):
    path, data = published
    data["measures"].append(copy.deepcopy(data["measures"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        player_input(data, "Peter", path)
    path.write_text('<script id="ex-data">{}</script><script id="ex-data">{}</script>')
    with pytest.raises(ValueError, match="Expected one"):
        read_payload(path)


def test_report_rejects_invented_evidence_cross_stroke_evidence_and_duplicate_strokes(published):
    path, data = published
    inputs = player_input(data, "Peter", path)
    valid = answer_for(inputs)
    validate_answer(valid, inputs)
    invented = copy.deepcopy(valid)
    invented["strokes"][0]["points"][0]["evidence_ids"] = ["serve.real_3d_angle"]
    with pytest.raises(ValidationError):
        validate_answer(invented, inputs)
    cross = copy.deepcopy(valid)
    cross["strokes"][0]["evidence_ids"] = ["backhand.swing_time"]
    with pytest.raises(ValueError, match="belong to serve"):
        validate_answer(cross, inputs)
    duplicate = copy.deepcopy(valid)
    duplicate["strokes"][0] = copy.deepcopy(duplicate["strokes"][1])
    with pytest.raises(ValueError, match="exactly once"):
        validate_answer(duplicate, inputs)


def test_html_escapes_model_text_and_uses_local_video_paths(published):
    path, data = published
    inputs = player_input(data, "Peter", path)
    answer = answer_for(inputs)
    answer["strokes"][0]["headline"] = '<img src=x onerror="alert(1)">'
    page = render_report([{"input": inputs, "answer": answer}], "../published-media/")
    assert '<img src=x' not in page and '&lt;img src=x' in page
    assert '../published-media/serve_Peter.mp4' in page
    assert '原始录像重新计算' in page and '有效击球数：未知' in page
    with pytest.raises(ValueError, match="Unsafe media base"):
        render_report([{"input": inputs, "answer": answer}], "javascript:alert(1)")
