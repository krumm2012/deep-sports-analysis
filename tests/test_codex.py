"""The Codex boundary: image forwarding, schema validation, caching and failures."""
import json
from pathlib import Path
import subprocess

import pytest

from dsa import codex


SCHEMA = {"type": "object", "properties": {"color": {"type": "string"}},
          "required": ["color"], "additionalProperties": False}


def test_image_request_is_validated_and_cached(tmp_path, monkeypatch):
    image = tmp_path / "frame.png"
    image.write_bytes(b"image1")
    calls = []
    monkeypatch.setattr(codex.shutil, "which", lambda _: "/bin/codex")

    def execute(cmd, **kw):
        calls.append(cmd)
        assert cmd[cmd.index("--image") + 1] == str(image.resolve())
        assert cmd[cmd.index("--sandbox") + 1] == "read-only"
        assert "--output-schema" in cmd and kw["input"].endswith("Color?")
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text('{"color":"red"}')
        return subprocess.CompletedProcess(cmd, 0, '{"type":"turn.completed","usage":{"input_tokens":7}}', "")

    monkeypatch.setattr(codex.subprocess, "run", execute)
    cache = tmp_path / "cache"
    answer, _ = codex.ask("Color?", [image], SCHEMA, cache)
    assert answer == {"color": "red"}
    assert codex.ask("Color?", [image], SCHEMA, cache) == (answer, 0.0)
    assert len(calls) == 1
    assert json.loads(next(cache.glob("*.usage.json")).read_text())["input_tokens"] == 7
    image.write_bytes(b"image2")
    codex.ask("Color?", [image], SCHEMA, cache)
    assert len(calls) == 2


def test_invalid_output_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(codex.shutil, "which", lambda _: "/bin/codex")
    calls = []

    def execute(cmd, **kw):
        calls.append(cmd)
        Path(cmd[cmd.index("--output-last-message") + 1]).write_text('{"color":42}')
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(codex.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="schema-valid"):
        codex.ask("Color?", [], SCHEMA, tmp_path / "cache", retries=1)
    assert len(calls) == 2 and not (tmp_path / "cache").exists()


def test_process_failure_is_not_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(codex.shutil, "which", lambda _: "/bin/codex")
    calls = []

    def execute(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "Login required")

    monkeypatch.setattr(codex.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="Login required"):
        codex.ask("Color?", [], SCHEMA, tmp_path / "cache")
    assert len(calls) == 1
