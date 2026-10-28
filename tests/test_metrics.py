import pytest

from m2rs.metrics import Scoreboard, match_positions, prf


def test_exact_match():
    tp, fp, fn = match_positions([2, 4], [2, 4], tol=0)
    assert (tp, fp, fn) == (2, 0, 0)


def test_tolerance_changes_matching():
    assert match_positions([3], [2], tol=0) == (0, 1, 1)
    assert match_positions([3], [2], tol=1) == (1, 0, 0)


def test_one_to_one_matching():
    # A single predicted boundary cannot cover two reference boundaries.
    tp, fp, fn = match_positions([4], [3, 4], tol=1)
    assert tp == 1 and fn == 1


def test_prf_edges():
    assert prf(0, 0, 0) == {'tp': 0, 'fp': 0, 'fn': 0, 'P': 0.0, 'R': 0.0,
                            'F1': 0.0}
    m = prf(3, 1, 1)
    assert m['P'] == pytest.approx(0.75)
    assert m['R'] == pytest.approx(0.75)


def test_micro_and_macro_differ():
    board = Scoreboard()
    board.add('small', 5, 0, 0)      # perfect, few boundaries
    board.add('large', 50, 50, 0)    # half precision, many boundaries
    s = board.summary()
    # micro weights by count: 55 / 105
    assert s['micro']['P'] == pytest.approx(55 / 105)
    # macro gives each protocol equal weight
    assert s['macro']['P'] == pytest.approx((1.0 + 0.5) / 2)
    assert s['micro']['P'] != pytest.approx(s['macro']['P'])
