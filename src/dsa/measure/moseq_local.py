"""Persistent keypoint-MoSeq fitting on an existing GPU, in its own environment.

Inputs are tracked, normalized keypoints, not video or unassociated detections.
This module imports JAX/keypoint-MoSeq only when fitting, so input validation and
diagnostics remain usable in the vision environment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np


def read_input(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    required = {"coordinates", "confidences", "bodyparts", "fps", "frame_indices"}
    if missing := required - data.keys():
        raise ValueError(f"Missing NPZ fields: {sorted(missing)}")
    xy = np.asarray(data["coordinates"], dtype=np.float64)
    conf = np.asarray(data["confidences"], dtype=np.float64)
    names = np.asarray(data["bodyparts"]).astype(str).tolist()
    if xy.ndim != 3 or xy.shape[-1] != 2 or xy.shape[0] < 20:
        raise ValueError("coordinates must have shape (T>=20, K, 2)")
    if conf.shape != xy.shape[:2] or len(names) != xy.shape[1] or len(set(names)) != len(names):
        raise ValueError("Confidence/bodypart dimensions do not match coordinates")
    if np.isinf(xy).any() or not np.isfinite(conf).all() or (conf < 0).any() or (conf > 1).any():
        raise ValueError("Coordinates may contain NaN, but confidences must be finite in [0,1]")
    if (np.isfinite(xy).all(-1).sum(0) < 2).any():
        raise ValueError("Each keypoint needs at least two observed frames")
    fps = float(np.asarray(data["fps"]).item())
    indices = np.asarray(data["frame_indices"])
    if not np.isfinite(fps) or fps <= 0 or indices.shape != (len(xy),) or not np.isfinite(indices).all():
        raise ValueError("Invalid frame rate or frame indices")
    if (np.diff(indices) < 0).any():
        raise ValueError("Frame indices must be nondecreasing")
    if not {"nose", "l_hip", "r_hip"}.issubset(names):
        raise ValueError("Whole-body input needs nose, l_hip and r_hip")
    return {**data, "coordinates": xy, "confidences": conf, "bodyparts": names, "fps": fps}


def diagnose(syllable: np.ndarray, fps: float) -> dict:
    labels = np.asarray(syllable)
    if labels.ndim != 1 or not len(labels) or not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
        raise ValueError("Syllables must be a nonempty nonnegative integer sequence")
    boundaries = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    lengths = np.diff(boundaries)
    ids, counts = np.unique(labels, return_counts=True)
    return {"frames": len(labels), "used_states": len(ids),
            "state_frames": {str(int(i)): int(n) for i, n in zip(ids, counts)},
            "dominant_state_fraction": float(counts.max() / len(labels)),
            "transition_count": len(lengths) - 1,
            "median_duration_seconds": float(np.median(lengths) / fps),
            "duration_percentiles_seconds": (np.percentile(lengths, [10, 50, 90]) / fps).tolist(),
            "fraction_runs_under_0_1_seconds": float((lengths / fps < .1).mean()),
            "runs": [{"state": int(labels[a]), "start_frame": int(a), "end_frame": int(b),
                      "start_seconds": float(a / fps), "end_seconds": float(b / fps)}
                     for a, b in zip(boundaries[:-1], boundaries[1:])],
            "collapsed": bool(len(ids) < 2),
            "label_order": "Raw model state IDs; neither semantic names nor frequency ranking"}


def _json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run(input_path: Path, output: Path, *, seed: int = 0, num_states: int = 5,
        latent_dim: int = 4, ar_iters: int = 100, full_iters: int = 200,
        kappa_ar: float = 1e6, kappa_full: float = 1e5,
        allow_cpu: bool = False) -> dict:
    for name, value in {"num_states": num_states, "latent_dim": latent_dim,
                        "ar_iters": ar_iters, "full_iters": full_iters}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not all(np.isfinite(v) and v > 0 for v in (kappa_ar, kappa_full)):
        raise ValueError("Kappas must be finite and positive")
    source = read_input(input_path)
    if (output / "config.yml").exists() or (output / "results.npz").exists():
        raise FileExistsError(f"Use a new output directory to preserve existing fits: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summary = {"status": "running", "input": str(input_path.resolve()),
               "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
               "started_at_utc": datetime.now(timezone.utc).isoformat(),
               "config": {"seed": seed, "num_states": num_states, "latent_dim_requested": latent_dim,
                          "ar_iters": ar_iters, "full_iters": full_iters,
                          "kappa_ar": kappa_ar, "kappa_full": kappa_full, "fps": source["fps"]},
               "limitations": ["Single short recording: in-sample segmentation, not validated generalization.",
                               "State IDs have no built-in tennis semantics or quality scores."]}
    _json(output / "summary.json", summary)
    try:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("MPLBACKEND", "Agg")
        from dsa.paths import ROOT

        os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(ROOT / "cache/jax-moseq"))
        import jax
        import keypoint_moseq as kpms
        from keypoint_moseq import fitting
        import h5py

        devices = jax.devices()
        if not allow_cpu and not any(d.platform == "gpu" for d in devices):
            raise RuntimeError(f"GPU required; detected JAX devices: {devices}")
        summary["backend"] = {"default": jax.default_backend(), "devices": [str(d) for d in devices],
                              "jax": jax.__version__, "keypoint_moseq": importlib.metadata.version("keypoint-moseq"),
                              "numpy": np.__version__, "allow_cpu": allow_cpu,
                              "compilation_cache": os.environ["JAX_COMPILATION_CACHE_DIR"]}
        names = source["bodyparts"]
        skeleton = [[a, b] for a, b in [("l_shoulder", "r_shoulder"), ("l_hip", "r_hip"),
                     ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"),
                     ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
                     ("l_hip", "l_knee"), ("l_knee", "l_ankle"),
                     ("r_hip", "r_knee"), ("r_knee", "r_ankle")] if a in names and b in names]
        # This new run directory already contains summary.json. We rejected any
        # existing config/results above; overwrite only initializes its config.
        kpms.setup_project(str(output), overwrite=True, bodyparts=names, use_bodyparts=names,
                           anterior_bodyparts=["nose"], posterior_bodyparts=["l_hip", "r_hip"],
                           skeleton=skeleton, fps=source["fps"])
        kpms.update_config(str(output), seed=seed, num_states=num_states, latent_dim=latent_dim, fix_heading=True)
        config = kpms.load_config(str(output))
        coords, confs = {"clip": source["coordinates"]}, {"clip": source["confidences"]}
        # format_data adds small regularizing noise; pin NumPy in addition to the JAX seed.
        np.random.seed(seed)
        data, metadata = kpms.format_data(coords, confs, **config)
        pca = kpms.fit_pca(**data, **config)
        kpms.save_pca(pca, str(output))
        used_dim = min(latent_dim, int(pca.n_components_))
        kpms.update_config(str(output), latent_dim=used_dim)
        config = kpms.load_config(str(output))
        # v0.6.6's config generator drops unknown fields such as seed.
        # Pass the sampler seed explicitly instead of relying on update_config.
        model = kpms.init_model(data, pca=pca, **{**config, "seed": seed})
        summary["initial_model_prng_key"] = np.asarray(jax.device_get(model["seed"])).tolist()
        summary["config"]["latent_dim"] = used_dim
        summary["pca_explained_variance"] = float(pca.explained_variance_ratio_[:used_dim].sum())
        summary["timing"] = {"setup_seconds": time.perf_counter() - started}
        summary["iteration_seconds"] = {"ar": [], "full": []}
        phase = ["ar"]
        original = fitting._wrapped_resample

        def timed_resample(*args, **kwargs):
            t = time.perf_counter()
            result = original(*args, **kwargs)
            jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, result)
            summary["iteration_seconds"][phase[0]].append(time.perf_counter() - t)
            return result

        def extract(model, name, filename):
            result = kpms.extract_results(model, metadata, str(output), name, save_results=False)["clip"]
            mapped = {"syllable": np.asarray(result["syllable"], dtype=np.int16),
                      "latent_state": np.asarray(result["latent_state"], dtype=np.float32),
                      "centroid": np.asarray(result["centroid"], dtype=np.float32),
                      "heading": np.asarray(result["heading"], dtype=np.float32)}
            if any(len(v) != len(source["coordinates"]) for v in mapped.values()):
                raise ValueError("MoSeq output length differs from source frame count")
            if not all(np.isfinite(v).all() for v in mapped.values()):
                raise ValueError("Nonfinite model output")
            np.savez_compressed(output / filename, **mapped, fps=source["fps"],
                                frame_indices=source["frame_indices"], bodyparts=np.asarray(names))
            return diagnose(mapped["syllable"], source["fps"])

        def require_snapshot(name, iteration):
            with h5py.File(output / name / "checkpoint.h5") as f:
                if str(iteration) not in f["model_snapshots"]:
                    raise RuntimeError(f"Fitting stopped before requested iteration {iteration}")

        fitting._wrapped_resample = timed_resample
        model_name = "model"
        try:
            t = time.perf_counter()
            model = kpms.update_hypparams(model, kappa=kappa_ar)
            model, model_name = kpms.fit_model(model, data, metadata, str(output), model_name,
                                              ar_only=True, num_iters=ar_iters,
                                              generate_progress_plots=False, save_every_n_iters=25)
            summary["timing"]["ar_seconds_including_jit"] = time.perf_counter() - t
            require_snapshot(model_name, ar_iters)
            summary["ar_diagnostics"] = extract(model, model_name, "ar_results.npz")
            _json(output / "summary.json", summary)
            phase[0] = "full"
            t = time.perf_counter()
            model = kpms.update_hypparams(model, kappa=kappa_full)
            model, model_name = kpms.fit_model(model, data, metadata, str(output), model_name,
                                              ar_only=False, start_iter=ar_iters,
                                              num_iters=ar_iters + full_iters,
                                              generate_progress_plots=False, save_every_n_iters=25)
            summary["timing"]["full_seconds_including_jit"] = time.perf_counter() - t
            require_snapshot(model_name, ar_iters + full_iters)
        finally:
            fitting._wrapped_resample = original
        kpms.extract_results(model, metadata, str(output), model_name)
        summary["diagnostics"] = extract(model, model_name, "results.npz")
        summary["checkpoint"] = str((output / model_name / "checkpoint.h5").resolve())
        summary["results_h5"] = str((output / model_name / "results.h5").resolve())
        summary["status"] = "complete"
        summary["timing"]["total_seconds"] = time.perf_counter() - started
        summary["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter() - started)
        _json(output / "summary.json", summary)
        (output / "failure.txt").write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-states", type=int, default=5)
    parser.add_argument("--latent-dim", type=int, default=4)
    parser.add_argument("--ar-iters", type=int, default=100)
    parser.add_argument("--full-iters", type=int, default=200)
    parser.add_argument("--kappa-ar", type=float, default=1e6)
    parser.add_argument("--kappa-full", type=float, default=1e5)
    parser.add_argument("--allow-cpu", action="store_true", help="Explicit CPU comparison only; never a silent fallback")
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, seed=args.seed, num_states=args.num_states,
                         latent_dim=args.latent_dim, ar_iters=args.ar_iters, full_iters=args.full_iters,
                         kappa_ar=args.kappa_ar, kappa_full=args.kappa_full, allow_cpu=args.allow_cpu),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
