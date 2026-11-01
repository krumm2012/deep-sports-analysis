# Deep Sports Analysis

Tennis video (phone or TV) → the two players over time (Player 1 / Player 2, near or far end), their
full skeletons, the ball, and their hits. Built only from off-the-shelf models (RF-DETR, WASB, RTMW,
TennisCourtDetector, F3ED) plus Codex CLI for marking the court. How it works, costs and status:
[docs/PIPELINE.md](docs/PIPELINE.md).

The skeleton quiz (Skeleton Tennis, https://offpolicyinc.com/work/skeleton-tennis/) is built from this pipeline; where its
files are and how it was made: [docs/QUIZ.md](docs/QUIZ.md).

One video, step by step (atp_hardcourt in full, with GPUs and costs: [docs/ATP_PIPELINE.md](docs/ATP_PIPELINE.md)):

```
            package   step                      runs on       writes (projects/<project>/<video>/)
            ───────   ────                      ───────       ────────────────────────────────────
 YouTube ─► ingest    download, slice           laptop        source.mp4  (whole upload: raw/<project>/)
            gpu       0  cut: live play only    Modal T4      kept.mp4, kept.json
            gpu       2  models                 Modal L40S    raw.npz, strokes/
            scene     3  players at each end    laptop        players.json
            scene     4  names                  laptop, Codex  names.json
            pages        overlay renders        laptop        render.mp4, render_plain.mp4
            gpu       2b 50/60 fps skeletons    Modal L40S    fine.npz
            measure   5  shots, strokes, moves  laptop        <project>/shots.json, technique/, movement/
            measure   6  compare players        laptop        <project>/profiles/
            pages     7  player pages           laptop        <project>/review/

 batch.py runs every video of a project through steps 0-4 and 2b, each step as soon as its input exists.
```

```
src/dsa/
  ingest/    projects.py (the catalog: projects, videos, players), download.py (YouTube / local -> source.mp4)
  gpu/       everything that runs on a Modal GPU, never on the laptop
             app.py (the Modal app: cut, analyse, fine_pose), process.py (step 2's frame loop), models.py
             (RF-DETR, WASB, RTMW-x, TennisCourtDetector), strokes.py (F3ED), cuts.py (step 0, + cuts.html),
             fine.py (step 2b), volume.py (the Modal volume and image)
  scene/     steps 3-4 from raw.npz: run.py (entry), tracks, court, court_snap, players, identity, events,
             clean, names (Codex), results.py (a video's results, read once, and its points)
  measure/   step 5-6: shots, technique, movement, syllables (Modal T4), profiles (players compared)
  pages/     what people look at: render (overlay videos), quiz, technique_report, movement_report
             (+ movement_explorer.js), clips, exemplars, serve (local server)
  batch.py   every video of a project through every step
  paths.py   where everything lives, the same on the laptop and on Modal (docs/STORAGE.md)
  video.py, codex.py   decoding and cutting; cached Codex CLI calls
tests/       one file per package, on made-up data (no models, no drive): uv run pytest
```

Setup: `uv sync --extra cloud --extra dev`; install Codex CLI and run `codex login` on the
CPU worker. Claude/Anthropic API is not used. Optional: `DSA_CODEX_MODEL`, `DSA_CODEX_BIN`,
`DSA_CODEX_TIMEOUT` (seconds, default 600). See [docs/CODEX_5090.md](docs/CODEX_5090.md). Data lives
under `data/` (the external drive) or `DSA_DATA`; layout and Modal in [docs/STORAGE.md](docs/STORAGE.md).

## Mac workbench

For a fresh checkout, create the lightweight CPU environment first (FFmpeg/ffprobe
and an existing GPU SSH alias are separate prerequisites):

```bash
uv venv .venv-workbench --python 3.14
uv pip install --python .venv-workbench/bin/python -r deploy/requirements-workbench-mac.txt
```

Start the local fixed-camera annotation and semantic training workspace:

```bash
bash deploy/run-workbench-mac.sh start
# http://127.0.0.1:8777/
bash deploy/run-workbench-mac.sh status
bash deploy/run-workbench-mac.sh restart
bash deploy/run-workbench-mac.sh stop
```

The first start deploys a runtime under `~/Library/Application Support/DeepSportsWorkbench`
and installs a per-user LaunchAgent. Existing source dataset files are linked into the
runtime without duplicating video bytes; new annotations and outputs are separate files.
The Git checkout remains the development source. Runtime annotations live in SQLite with
immutable JSON revision backups, independently of browser storage. Existing model/data
artifacts are provisioned locally and are not included in Git.

The workspace connects complete-clip annotation, CPU supervised training, versioned model
reports, and JSON prediction. New fixed-camera MP4 clips (up to 120 seconds / 500MB) can be
standardized locally, processed by the optional 5090 worker, and returned for inference.
Imports remain outside the frozen train/validation dataset. Each project permits one
background task at a time, with logs and retry using preserved inputs. MPS/CoreML video
inference is not implemented yet.

The optional GPU settings are in `config/workbench-backend.json`; they use an existing SSH
alias without copying credentials. The UI and API listen only on loopback. Mutating API
requests require `X-Workbench-Request: 1` and reject other web origins. For example:

```bash
curl -X POST http://127.0.0.1:8777/api/projects/camera05_custom/predict \
  -H 'Content-Type: application/json' -H 'X-Workbench-Request: 1' \
  -d '{"clip_id":"18_08","stroke_type":"无法判断"}'
```

Deployment verification and model details: [docs/CAMERA05_POSE_V5.md](docs/CAMERA05_POSE_V5.md).
The lightweight CPU environment is specified in `deploy/requirements-workbench-mac.txt`;
the broader video pipeline still uses its separate vision / MoSeq environments.

License: MIT ([LICENSE](LICENSE)), for this repository's code. The models and code it downloads and runs
(RF-DETR, WASB, RTMW, TennisCourtDetector, F3Set / F3ED, keypoint-MoSeq) keep their own licenses.
