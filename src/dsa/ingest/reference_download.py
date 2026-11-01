"""Resumable, sequential yt-dlp downloads of the reproduction's source matches.

The manifest identifies uploads, not evidence that every historical source is public.
Network proxy access may be provided with --proxy; no browser credentials are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from dsa.ingest.published import inspect_video


def run(manifest: Path, root: Path, proxy: str | None = None, limit: int = 0) -> dict:
    matches = json.loads(manifest.read_text())["matches"]
    if limit:
        matches = matches[:limit]
    state_path = root / "reference-download-status.json"
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    results = {m["youtube_id"]: m for m in previous.get("results", [])}
    logs = root / "download-logs"
    logs.mkdir(parents=True, exist_ok=True)

    def save(status):
        state = {"manifest": str(manifest.resolve()), "expected": len(matches), "status": status,
                 "updated_at_utc": datetime.now(timezone.utc).isoformat(), "results": list(results.values()),
                 "verified": sum(v["status"] == "verified" for v in results.values()),
                 "failed": sum(v["status"] == "failed" for v in results.values())}
        tmp = state_path.with_suffix(".json.part")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        tmp.replace(state_path)
        return state

    consecutive_failures = 0
    for index, match in enumerate(matches):
        yt = match["youtube_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", yt):
            raise ValueError(f"Invalid YouTube ID: {yt}")
        folder = root / ("own_public_2025" if match["catalog"] == "own_public_2025" else "atp_hardcourt")
        folder.mkdir(parents=True, exist_ok=True)
        path, log = folder / f"{yt}.mp4", logs / f"{yt}.log"
        row = {**match, "source_url": f"https://www.youtube.com/watch?v={yt}",
               "file": str(path.resolve()), "log": str(log.resolve()), "status": "downloading"}
        results[yt] = row
        save("running")
        print(f"{index + 1}/{len(matches)} {yt} {match['match']}", flush=True)
        try:
            if shutil.disk_usage(root).free < 20 * 1024 ** 3:
                raise RuntimeError("Less than 20 GiB free; download stopped")
            if not path.exists():
                cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--no-warnings", "--newline",
                       "--socket-timeout", "20", "--retries", "2", "--extractor-retries", "1",
                       "--fragment-retries", "3", "--js-runtimes", "node", "--remote-components", "ejs:github",
                       "-N", "8", "-f", "bv*[height<=720][ext=mp4]", "--write-info-json",
                       "--max-filesize", "12G", "-o", str(path), row["source_url"]]
                if proxy:
                    cmd += ["--proxy", proxy]
                if shutil.which("aria2c"):
                    # Still one video at a time; parallel ranges speed up the
                    # single HTTP stream that -N (DASH fragments) cannot split.
                    cmd += ["--downloader", "aria2c", "--downloader-args", "aria2c:-x8 -s8 -k10M"]
                with log.open("a") as stream:
                    subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=7200)
            metadata = inspect_video(path)
            info_file = path.with_suffix(".info.json")
            if info_file.exists():
                info = json.loads(info_file.read_text())
                row["youtube_metadata"] = {k: info.get(k) for k in
                                           ("id", "title", "channel", "channel_id", "upload_date", "duration", "fps", "format_id")}
            row.update(status="verified", **metadata)
            consecutive_failures = 0
        except (subprocess.SubprocessError, OSError, ValueError, KeyError, RuntimeError) as exc:
            row.update(status="failed", error=str(exc))
            consecutive_failures += 1
        save("running")
        print(f"  {row['status']} {row.get('bytes', 0)} bytes", flush=True)
        if consecutive_failures >= 3:
            return save("stopped_after_three_consecutive_failures")
    return save("complete" if all(results[m["youtube_id"]]["status"] == "verified" for m in matches) else "incomplete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--proxy")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    result = run(args.manifest, args.root, args.proxy, args.limit)
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
