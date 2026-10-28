"""Boundary matching and P/R/F1 (Section 5.1)."""

from typing import Iterable, List, Sequence, Tuple


def match_positions(pred: Iterable[int], gt: Sequence[int],
                    tol: int = 0) -> Tuple[int, int, int]:
    """Greedy one-to-one matching within +-``tol`` bytes.

    Returns (tp, fp, fn).  With tol=0 a prediction hits a reference boundary
    only when the byte offsets coincide, which is the setting used in the
    paper.
    """
    pred = sorted(pred)
    used = set()
    tp = 0
    for p in pred:
        best = None
        for gi, g in enumerate(gt):
            if gi in used:
                continue
            d = abs(g - p)
            if d <= tol and (best is None or d < best[0]):
                best = (d, gi)
        if best is not None:
            used.add(best[1])
            tp += 1
    return tp, len(pred) - tp, len(gt) - len(used)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'P': p, 'R': r, 'F1': f1}


class Scoreboard:
    """Accumulates counts per protocol and in total."""

    def __init__(self):
        self.per_proto: dict = {}
        self.total = [0, 0, 0]

    def add(self, proto: str, tp: int, fp: int, fn: int):
        acc = self.per_proto.setdefault(proto, [0, 0, 0])
        acc[0] += tp
        acc[1] += fp
        acc[2] += fn
        self.total[0] += tp
        self.total[1] += fp
        self.total[2] += fn

    def summary(self) -> dict:
        micro = prf(*self.total)
        per = {k: prf(*v) for k, v in sorted(self.per_proto.items())}
        macro = {}
        if per:
            for k in ('P', 'R', 'F1'):
                macro[k] = sum(v[k] for v in per.values()) / len(per)
        return {'per_protocol': per, 'micro': micro, 'macro': macro}

    def format(self) -> List[str]:
        rows = []
        for name, v in sorted(self.per_proto.items()):
            m = prf(*v)
            rows.append('%-10s TP=%-6d FP=%-6d FN=%-6d P=%.4f R=%.4f F1=%.4f'
                        % (name, v[0], v[1], v[2], m['P'], m['R'], m['F1']))
        m = prf(*self.total)
        rows.append('-' * 70)
        rows.append('%-10s TP=%-6d FP=%-6d FN=%-6d P=%.4f R=%.4f F1=%.4f'
                    % ('micro', *self.total, m['P'], m['R'], m['F1']))
        return rows
