"""Ablation study of Section 5.2.3.

Each variant disables one mechanism or one feature family through the engine
switches and is evaluated on the same test captures.

    python scripts/ablation.py            # full method + mechanisms
    python scripts/ablation.py --features # feature families as well
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2rs.cli import evaluate, load_config  # noqa: E402

MECHANISMS = [
    ('w/o length stratification', {'use_stratify': False}),
    ('w/o discriminative scoring', {'use_scorer': False}),
    ('w/o multi-scale fusion', {'use_fusion': False}),
    ('w/o width prior', {'use_prior': False}),
    ('w/o MAP decoding', {'use_map': False}),
]

FAMILIES = [
    ('w/o semantic-region features', {'drop_features': 'semantic'}),
    ('w/o length/sequence features', {'drop_features': 'length_seq'}),
    ('w/o positional-prior features', {'drop_features': 'positional'}),
    ('w/o neighbour-context features', {'drop_features': 'neighbour'}),
]


def run_variant(cfg: dict, over: dict, tol: int, model_dir):
    c = {**cfg, 'boundary': {'m2rs': {**cfg['boundary']['m2rs'], **over}}}
    return evaluate(ROOT / 'data' / 'test_protocol', c, tol, model_dir,
                    verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tol', type=int, default=0)
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--features', action='store_true')
    a = ap.parse_args()

    cfg = load_config()
    base = run_variant(cfg, {}, a.tol, a.model_dir)['micro']
    print('%-32s P=%.4f R=%.4f F1=%.4f' % ('full M2RS', base['P'],
                                            base['R'], base['F1']))
    variants = MECHANISMS + (FAMILIES if a.features else [])
    for name, over in variants:
        m = run_variant(cfg, over, a.tol, a.model_dir)['micro']
        print('%-32s P=%.4f R=%.4f F1=%.4f  dF1=%+.4f'
              % (name, m['P'], m['R'], m['F1'], m['F1'] - base['F1']))


if __name__ == '__main__':
    main()
