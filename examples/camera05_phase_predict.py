"""Predict camera05 phases from a prepared 25fps body-pose NPZ."""
import argparse
import json
from pathlib import Path

from dsa.measure.semantic_inference import predict
from dsa.measure.semantic_strokes import STROKE_TYPES


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
