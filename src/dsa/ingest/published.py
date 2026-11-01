"""Archive the article's public video clips with hashes and decoded metadata.

This downloads published, already edited examples, not the private full recordings.
    python -m dsa.ingest.published snapshot.html data/published-media
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


ARTICLE = "https://offpolicyinc.com/work/claude-tennis-coach/"


def urls_from_page(text: str) -> list[str]:
    text = html.unescape(text).replace(r"\/", "/")
    urls = set(re.findall(r'https://media\.offpolicyinc\.com/claude-tennis-coach/[^\s"<>\\]+?\.mp4', text))
    for url in urls:
        name = Path(urlsplit(url).path).name
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.mp4", name):
            raise ValueError(f"Unexpected media filename: {name}")
    return sorted(urls)


def inspect_video(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, timeout=45)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    if stream["width"] <= 0 or stream["height"] <= 0 or float(info["format"]["duration"]) <= 0:
        raise ValueError(f"Empty video: {path}")
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return {**stream, "duration_seconds": float(info["format"]["duration"]),
            "bytes": path.stat().st_size, "sha256": sha.hexdigest()}


def download(url: str, dest: Path) -> dict:
    path = dest / Path(urlsplit(url).path).name
    if path.exists():
        return {"url": url, "file": path.name, "status": "verified", **inspect_video(path)}
    part = path.with_suffix(".mp4.part")
    for attempt in range(3):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"User-Agent": "deep-sports-analysis-reproduction/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(url, headers=headers), timeout=45) as response:
                resume = offset > 0 and response.status == 206
                expected = int(response.headers.get("Content-Length", "0"))
                written = 0
                with part.open("ab" if resume else "wb") as f:
                    while block := response.read(256 * 1024):
                        f.write(block)
                        written += len(block)
                if expected and written != expected:
                    raise IOError(f"Incomplete download: {written}/{expected}")
            info = inspect_video(part)
            part.replace(path)
            return {"url": url, "file": path.name, "status": "verified", **info}
        except Exception as exc:
            if isinstance(exc, HTTPError) and exc.code == 416:
                part.unlink(missing_ok=True)
            if attempt == 2:
                return {"url": url, "file": path.name, "status": "failed", "error": str(exc)}
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def archive(snapshot: Path, dest: Path, workers: int = 4) -> dict:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    raw = snapshot.read_bytes()
    urls = urls_from_page(raw.decode())
    if not urls:
        raise ValueError("No public video URLs found")
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"article": ARTICLE, "source_snapshot_sha256": hashlib.sha256(raw).hexdigest(),
                "provenance": "Published edited example clips; not full private source recordings or new GPU outputs",
                "expected": len(urls), "files": []}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(download, url, dest): url for url in urls}
        for future in as_completed(tasks):
            result = future.result()
            manifest["files"].append(result)
            manifest["verified"] = sum(f["status"] == "verified" for f in manifest["files"])
            manifest["failed"] = sum(f["status"] == "failed" for f in manifest["files"])
            manifest["bytes"] = sum(f.get("bytes", 0) for f in manifest["files"])
            manifest["elapsed_seconds"] = round(time.monotonic() - started, 1)
            tmp = dest / "manifest.json.part"
            tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            tmp.replace(dest / "manifest.json")
            print(f"{len(manifest['files'])}/{len(urls)} {result['status']} {result['file']}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = archive(args.snapshot, args.dest, args.workers)
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
