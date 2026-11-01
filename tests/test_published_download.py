"""An interrupted archive must resume without corrupting its final file."""
from io import BytesIO

from dsa.ingest import published


def test_media_inventory_covers_json_urls_and_deduplicates():
    url = "https://media.offpolicyinc.com/claude-tennis-coach/clip.mp4"
    escaped = url.replace("/", r"\/")
    page = f'<video src="{url}"></video><script>{{"video":"{escaped}"}}</script>'
    page += '<video src="https://unrelated.example/private.mp4"></video>'
    assert published.urls_from_page(page) == [url]


def test_server_ignoring_range_replaces_partial_instead_of_appending(tmp_path, monkeypatch):
    (tmp_path / "clip.mp4.part").write_bytes(b"old-partial")
    calls = []

    class Response(BytesIO):
        status = 200  # Server returns the whole object despite the Range request.
        headers = {"Content-Length": "9"}

    def request(req, timeout):
        calls.append(req.headers)
        return Response(b"new-video")

    def inspect(path):
        assert path.read_bytes() == b"new-video"
        return {"bytes": 9}

    monkeypatch.setattr(published, "urlopen", request)
    monkeypatch.setattr(published, "inspect_video", inspect)
    result = published.download("https://media.offpolicyinc.com/claude-tennis-coach/clip.mp4", tmp_path)
    assert result["status"] == "verified"
    assert calls[0]["Range"] == "bytes=11-"
    assert (tmp_path / "clip.mp4").read_bytes() == b"new-video"
    assert not (tmp_path / "clip.mp4.part").exists()


def test_truncated_download_never_becomes_final(tmp_path, monkeypatch):
    class Response(BytesIO):
        status = 200
        headers = {"Content-Length": "20"}

    monkeypatch.setattr(published, "urlopen", lambda *a, **kw: Response(b"truncated"))
    monkeypatch.setattr(published.time, "sleep", lambda _: None)
    result = published.download("https://media.offpolicyinc.com/claude-tennis-coach/clip.mp4", tmp_path)
    assert result["status"] == "failed"
    assert not (tmp_path / "clip.mp4").exists()
