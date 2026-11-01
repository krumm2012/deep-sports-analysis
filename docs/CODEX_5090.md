# Codex and an existing NVIDIA GPU

The inference boundary is `dsa.codex.ask`: Codex CLI with image attachments,
JSON-schema output, full local schema validation and disk caching. No Claude or
Anthropic API call remains. Historical article text in the report templates is
kept as the original author's account, not relabelled as newly generated advice.

Authenticate the CPU worker with `codex login`. Authentication stays on that
machine; do not copy the Mac's Codex credentials to the GPU server. Scene work
can run on the Mac after retrieving `raw.npz` and `strokes/` from the GPU server.

Optional environment variables:

- `DSA_CODEX_MODEL`: use an explicit model for reproducible runs; otherwise the CLI's configured default.
- `DSA_CODEX_BIN`: executable name or path (default `codex`).
- `DSA_CODEX_TIMEOUT`: per-inference timeout in seconds (default 600).
- `DSA_DATA`: data root, resolved to an absolute path at import.

An explicit model is recommended when sharing caches: the configured-default
cache does not automatically detect changes to the CLI's default model.

## GPU worker

Use `python -m dsa.gpu.local` on a verified CUDA machine. This entry point does
not import Modal or create a cloud job. Install dependencies in a project venv
(`uv sync --extra dev`), then verify CUDA:

```bash
uv run python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

Provision these upstream assets under the data root before running models:

```text
code/wasb/src/configs/model/wasb.yaml
code/wasb/pretrained_weights/wasb_tennis_best.pth.tar
code/f3set/                         # F3Set source; needed when strokes=true or TV
models/f3ed/config.json
models/f3ed/checkpoint_049.pt
code/tennis_court_detector/         # TV/cut selection only
models/tcd/model_tennis_court_det.pt
```

These files were not shipped in the upstream repository. RF-DETR and RTMW also
need their downloadable weights and outbound network access on first use. RTX
5090 compatibility of the installed CUDA/PyTorch/ONNX Runtime must be checked
on the host. The upstream cloud image uses `onnxruntime-gpu==1.30.0`,
`timm==0.6.13`, `tabulate==0.9.0`; install appropriate GPU dependencies in the
worker venv. Avoid CPU `onnxruntime` shadowing the GPU provider (see
`dsa.gpu.volume.image`).

## Public demonstration input

The local working copy contains the site's 23-second, 1280x720, 30 fps phone
rally. It is demonstration footage, not the author's full private recording.
Re-fetch it if moving to another checkout:

```bash
mkdir -p data/raw/demo data/projects/demo/rally
curl -fL https://media.offpolicyinc.com/claude-tennis-coach/rally_peter_richard.mp4 \
  -o data/raw/demo/rally_peter_richard.mp4
cp data/raw/demo/rally_peter_richard.mp4 data/projects/demo/rally/source.mp4
export DSA_DATA="$PWD/data"
export DSA_PROJECTS="$PWD/config/demo-projects.json"
```

On the GPU worker, after provisioning weights, run the complete short clip:

```bash
uv run python -m dsa.gpu.local demo/rally
```

Then on the authenticated CPU worker, with the same input and GPU outputs:

```bash
uv run python -m dsa.scene.court demo/rally
uv run python -m dsa.scene.run demo/rally --court-marked
```

Peter/Richard identity must be assigned from known footage by a person; a model
cannot know private players' names from their appearance. A single short rally is insufficient for a reliable serve-technique assessment. Pro-comparison reports also
require processing the `atp_hardcourt` reference matches; those datasets are not
included. `dsa.batch` remains the upstream Modal batch scheduler; use the local
worker entry point above for 5090 runs.

For a limited GPU smoke test use `--seconds 10` (writes `test/`) and add `--test`
to scene processing; omit that flag for the full 23-second demo. `--cuts` runs
broadcast live-play selection first, on the same GPU.

## Verified deployment (2026-10-31, Asia/Shanghai)

- Host: `ubuntu32`, NVIDIA GeForce RTX 5090 D, 32607 MiB VRAM.
- Project: `/root/deep-sports-analysis`; project venv uses the existing read-only
  PyTorch 2.8.0+cu128 / torchvision 0.23.0 base via `5090-base.pth`.
- Use `deploy/requirements-5090.txt` and `deploy/constraints-5090.txt` for this
  deployment instead of syncing the cloud lockfile into the 5090 environment.
- ONNX Runtime GPU 1.23.2 is used for CUDA 12.8. The upstream 1.30 default wheel
  requires CUDA 13, so it is not the wheel deployed here. CPU ONNX Runtime was
  removed from the overlay; model initialization now rejects silent CPU fallback.
- Model hashes and upstream source revisions: `deploy/assets-5090.json`.
- Full input: 690 frames, 1280x720, 30 fps, 23 seconds. GPU job exit code: 0.
- Outputs: `raw.npz`, `strokes/scores.npz`, `strokes/strokes.json`.
- GPU pose pass: 3047 posed person crops; 371/690 ball detections; four predicted
  shot events. These are model outputs, not ground-truth accuracy measurements.
- Local and remote regression suites: 40 tests passed on both machines.
- Codex court marking: 14 points; CPU postprocessing: 511/690 frames with both
  players tracked. Anonymous IDs are retained; personal identities are not inferred.
- Final output directory: `data/projects/demo/rally/` (excluded from git).
  `render.mp4` and `render_plain.mp4` were decoded and verified as 690-frame,
  23-second H.264 videos. `index.html` is the report entry point.
- Verification logs and installed environment snapshot are retained under
  `verification/` in the remote project and under local `data/verification/`.
- SSH alias retains strict host-key verification and existing identity file;
  the proxy reuses a live AweSun native SSH tunnel after matching its pinned
  host key. A shared SSH control connection reduces relay setup churn. Keep the
  AweSun SSH tunnel available when starting a new SSH transport.
- graphify rebuild was attempted but the current local environment lacks graphify.

### Rerun

With the existing provisioned weights and source video on ubuntu32:

```bash
ssh ubuntu32-sam5090 'cd /root/deep-sports-analysis && bash deploy/run-demo-5090.sh'
```

Weights are local, so this demo no longer needs a download proxy. Completed GPU
outputs are reused. Do CPU postprocessing and Codex reporting on the authenticated
Mac, using `.venv/bin/python` with `PYTHONPATH=src` and the demo `DSA_PROJECTS` file.
Codex credentials were not copied to the remote host.

Codex reference: https://learn.chatgpt.com/docs/developer-commands#codex-exec
CUDA compatibility: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
