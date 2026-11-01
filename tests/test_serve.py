"""Browser-facing text encoding and binary range responses."""
from functools import partial
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.request import Request, urlopen

import pytest

from dsa.pages.serve import Ranges


@pytest.fixture
def server(tmp_path):
    (tmp_path / "prompt.txt").write_text("请用中文分析网球动作。", encoding="utf-8")
    (tmp_path / "input.json").write_text('{"球员":"Richard"}', encoding="utf-8")
    (tmp_path / "clip.mp4").write_bytes(bytes(range(20)))
    http = ThreadingHTTPServer(("127.0.0.1", 0), partial(Ranges, directory=str(tmp_path)))
    thread = Thread(target=http.serve_forever, kwargs={"poll_interval": .05}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}"
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def test_chinese_prompt_and_json_declare_their_encoding(server):
    for path, expected in [("prompt.txt", "请用中文分析网球动作。"), ("input.json", '{"球员":"Richard"}')]:
        with urlopen(f"{server}/{path}") as response:
            assert response.headers.get_content_charset() == "utf-8"
            assert response.read().decode(response.headers.get_content_charset()) == expected
    with urlopen(Request(f"{server}/prompt.txt", headers={"Range": "bytes=0-2"})) as response:
        assert response.status == 206
        assert response.headers.get_content_charset() == "utf-8"
        assert response.read().decode("utf-8") == "请"


def test_video_ranges_keep_binary_content_type(server):
    with urlopen(Request(f"{server}/clip.mp4", headers={"Range": "bytes=4-8"})) as response:
        assert response.status == 206
        assert response.headers["Content-Type"] == "video/mp4"
        assert response.headers["Content-Range"] == "bytes 4-8/20"
        assert response.read() == bytes(range(4, 9))
