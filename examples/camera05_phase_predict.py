"""Predict camera05 v5 phases from a prepared 25fps body-pose NPZ."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from dsa.measure.moseq_local import read_input
from dsa.measure.semantic_phases import PHASES
from dsa.measure.semantic_pilot import sha
from dsa.measure.semantic_pose import frame_features, runs
from dsa.measure.semantic_strokes import STROKE_TYPES
from dsa.measure.semantic_supervised import predict_frames


BODY_PARTS = ['nose', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
              'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee',
              'l_ankle', 'r_ankle', 'l_bigtoe', 'l_smalltoe', 'l_heel',
              'r_bigtoe', 'r_smalltoe', 'r_heel']


def predict(model_directory, input_path, stroke_type='无法判断'):
    summary = json.loads((model_directory/'summary.json').read_text())
    if summary.get('training_type') != 'fully_supervised' or summary['status'] != 'complete':
        raise ValueError('A completed supervised model is required')
    if sha(model_directory/'model.joblib') != summary['model_sha256']:
        raise ValueError('Model checksum differs from summary')
    data = read_input(input_path)
    if data['fps'] != 25 or data['bodyparts'] != BODY_PARTS:
        raise ValueError('Requires 25fps and the exact camera05 19-joint order')
    if not np.array_equal(data['frame_indices'], np.arange(len(data['coordinates']))):
        raise ValueError('Requires continuous zero-based 25fps frame indices')
    if stroke_type not in STROKE_TYPES:
        raise ValueError('Unknown stroke type')
    observed = np.asarray(data.get('raw_observed'))
    if observed.shape != data['coordinates'].shape[:2] or observed.dtype != bool:
        raise ValueError('raw_observed must be a (T,19) boolean observation mask')
    # Load only the project's trusted model artifact.
    model = joblib.load(model_directory/'model.joblib')
    pred, score, probability, accepted, distance = predict_frames(
        model, frame_features(data), stroke_type,
        summary['selection']['nearest_anchor_distance_limit'])
    segments = [{'start_frame': a, 'end_frame_exclusive': b,
                 'start_seconds': a/25, 'end_seconds': b/25,
                 'phase': PHASES[p] if p >= 0 else None,
                 'mean_score': float(score[a:b].mean())}
                for a, b, p in runs(np.where(accepted, pred, -1))]
    return {'version': summary['version'], 'model_sha256': summary['model_sha256'],
            'fps': 25, 'frames': len(pred), 'stroke_type_supplied': stroke_type,
            'core_expert_used': stroke_type if stroke_type in model.experts else None,
            'phase_order': list(PHASES), 'score_is_calibrated_probability': False,
            'accepted_fraction': float(accepted.mean()), 'segments': segments}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--stroke-type', choices=STROKE_TYPES, default='无法判断')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = predict(args.model_dir, args.input, args.stroke_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'frames': result['frames'],
                      'accepted_fraction': result['accepted_fraction']}, ensure_ascii=False))
