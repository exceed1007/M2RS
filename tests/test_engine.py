import math

import numpy as np

from m2rs.engine import (M2RSBoundaryEngine, WidthPrior, decode_greedy,
                         decode_map)
from m2rs.features import features_for_stratum


def test_features_are_finite_and_complete():
    M = np.array([[1, 2, 3, 0], [1, 2, 4, 0], [1, 2, 5, 0]], dtype=np.uint8)
    rows = features_for_stratum(M, 4)
    assert len(rows) == 3                       # j = 1 .. L-1
    assert all(math.isfinite(v) for v in rows[0].values())
    assert set(rows[0]) == set(rows[2])


def test_map_recovers_the_strongest_cut():
    prior = WidthPrior()
    cuts = decode_map(4, [12.0, -8.0, 12.0], prior, theta=0.0)
    assert 1 in cuts
    assert all(0 < b < 4 for b in cuts)


def test_lower_theta_yields_more_cuts():
    prior = WidthPrior()
    logits = [1.0, -1.5, 1.0, -1.5, 1.0, -1.5]
    strict = decode_map(7, logits, prior, theta=0.0)
    loose = decode_map(7, logits, prior, theta=-6.0)
    assert len(loose) >= len(strict)


def test_greedy_is_a_subset_of_the_same_signal():
    prior = WidthPrior()
    logits = [12.0, -8.0, 12.0]
    assert set(decode_greedy(4, logits, prior)) <= {1, 3}


def test_engine_returns_interior_boundaries():
    eng = M2RSBoundaryEngine()
    msgs = [bytes([1, 2, 3, 0, 0, 9]), bytes([1, 2, 4, 0, 0, 8]),
            bytes([1, 2, 5, 0, 0, 7])]
    preds = eng.predict_messages(msgs)
    assert len(preds) == len(msgs)
    for msg, cuts in zip(msgs, preds):
        assert all(0 < b < len(msg) for b in cuts)
        # equal-length messages share one template
        assert cuts == preds[0]


def test_stratification_ablation_runs():
    eng = M2RSBoundaryEngine({'boundary': {'m2rs': {'use_stratify': False}}})
    msgs = [bytes(range(8)), bytes(range(6))]
    preds = eng.predict_messages(msgs)
    assert len(preds) == 2
    assert all(0 < b < len(m) for m, p in zip(msgs, preds) for b in p)
