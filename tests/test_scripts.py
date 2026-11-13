"""Tests for the experiment helpers: no capture and no tshark needed."""

from pathlib import Path

import pytest

from scripts.common import FAMILY, pick_groups, prior_from_labels

ROOT = Path(__file__).resolve().parents[1]


def test_prior_matches_hand_computed_widths():
    # Stratum A: 5 bytes, one cut at 2  -> widths 2 (interior) and 3 (final).
    # Stratum B: 4 bytes, cuts at 1 and 3 -> 1, 2 (interior) and 1 (final).
    groups = ['A#5'] * 4 + ['B#4'] * 3
    y = [0, 1, 0, 0, 1, 0, 1]          # labels of positions 1 .. L-1
    prior = prior_from_labels(groups, y, max_w=8)

    p_mid, p_last = prior['p_mid'], prior['p_last']
    assert pytest.approx(sum(p_mid.values()), abs=1e-6) == 1.0
    assert pytest.approx(sum(p_last.values()), abs=1e-6) == 1.0
    # interior widths: 1 occurs once, 2 occurs twice, 3 never
    assert float(p_mid['2']) > float(p_mid['1']) > float(p_mid['3'])
    # final widths: 1 (stratum B) and 3 (stratum A)
    assert float(p_last['1']) == pytest.approx(float(p_last['3']), rel=1e-3)
    assert float(p_last['2']) < float(p_last['1'])


def test_prior_never_returns_zero_probability():
    prior = prior_from_labels(['A#6'] * 5, [1, 0, 0, 1, 0], max_w=16)
    assert all(v > 0 for v in prior['p_mid'].values())
    assert prior['max_len'] == 16


def test_pick_groups_is_by_stratum_and_reproducible():
    groups = ['s%d#10' % i for i in range(20)]
    half = pick_groups(groups, 0.5, seed=7)
    assert len(half) == 10
    assert set(half) <= set(groups)
    assert half == pick_groups(groups, 0.5, seed=7)     # same seed, same draw
    assert pick_groups(groups, 1.0, seed=7) != [] and \
        len(pick_groups(groups, 1.0, seed=7)) == 20
    assert pick_groups(groups, 0.0, seed=7) != []       # never an empty corpus


def test_every_bundled_capture_has_a_family():
    stems = {p.stem for p in (ROOT / 'data' / 'test_protocol').iterdir()
             if p.is_file()}
    assert stems, 'test captures are missing'
    missing = stems - set(FAMILY)
    assert not missing, 'no protocol family assigned to %s' % sorted(missing)
