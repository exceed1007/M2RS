"""Position-level structural features (Section 4.3).

The same functions are used for offline training and for inference, so the
feature definition lives in exactly one place.
"""

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

# Feature families used by the ablation study in Section 5.2.3.
FEATURE_FAMILIES = {
    'semantic': ['strl', 'strr', 'strboth', 'prun', 'prun_both', 'zrun',
                 'txt_rate'],
    'length_seq': ['selflen1', 'selflen2', 'totlen1', 'totlen2', 'idxcorr',
                   'cntinc', 'lencorr'],
    'positional': ['relpos', 'logpos', 'even2', 'even4', 'is2tok'],
    'neighbour': ['prev_cardchg', 'prev_hchg', 'prev_const', 'next_cardchg',
                  'next_hchg', 'next_const'],
}


def entropy_bits(counts: Sequence[int], n: int) -> float:
    """Empirical entropy in bits with Miller-Madow bias correction."""
    if n <= 0 or not counts:
        return 0.0
    h = 0.0
    m = 0
    for c in counts:
        if c <= 0:
            continue
        m += 1
        p = c / n
        h -= p * math.log2(p)
    return max(0.0, h + (m - 1) / (2.0 * n))


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    with np.errstate(invalid='ignore', divide='ignore'):
        c = float(np.corrcoef(x, y)[0, 1])
    return c if np.isfinite(c) else 0.0


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    ha = entropy_bits(list(Counter(a.tolist()).values()), n)
    hb = entropy_bits(list(Counter(b.tolist()).values()), n)
    hab = entropy_bits(list(Counter(zip(a.tolist(), b.tolist())).values()), n)
    return max(0.0, ha + hb - hab)


def col_stats(M: np.ndarray, j: int) -> Dict[str, float]:
    v = M[:, j]
    c = Counter(v.tolist())
    n = len(v)
    return {'card': len(c), 'H': entropy_bits(list(c.values()), n),
            'const': len(c) == 1,
            'zero': float(np.mean(v == 0)),
            'printable': float(np.mean([(32 <= b <= 126) for b in v]))}


def _run_length(mask_rows: np.ndarray) -> np.ndarray:
    """Length of the True run that each cell belongs to, per row."""
    n, L = mask_rows.shape
    run = np.zeros((n, L), dtype=np.int32)
    for r in range(n):
        i = 0
        while i < L:
            if mask_rows[r, i]:
                j = i
                while j + 1 < L and mask_rows[r, j + 1]:
                    j += 1
                run[r, i:j + 1] = j - i + 1
                i = j + 1
            else:
                i += 1
    return run


def features_for_stratum(M: np.ndarray, L: int,
                         row_len: Sequence[int] = None
                         ) -> List[Dict[str, float]]:
    """Features for every candidate cut point j = 1 .. L-1 of an n x L matrix.

    ``row_len`` gives the real message length of each row (used by the
    cross-stratum global view, where rows differ in length).  It defaults to
    ``L``, i.e. the stratum is treated as equal-length.
    """
    n = M.shape[0]
    cs = [col_stats(M, j) for j in range(L)]
    lens_arr = (np.asarray(row_len, dtype=np.int64) if row_len is not None
                else np.full(n, L, dtype=np.int64))
    idx_arr = np.arange(n, dtype=np.float64)
    isp = np.array([[(32 <= b <= 126) or b in (9, 10, 13) for b in row]
                    for row in M], dtype=bool) if n else np.zeros((0, L), bool)
    run = _run_length(isp)
    runz = _run_length(M == 0)
    rows: List[Dict[str, float]] = []
    for j in range(1, L):
        l, r = cs[j - 1], cs[j]
        lm1 = cs[j - 2] if j >= 2 else None
        rp1 = cs[j + 1] if j + 1 < L else None
        pair = M[:, j - 1:j + 1]
        cpair = len({tuple(x) for x in pair.tolist()})
        pair_int = np.array([int.from_bytes(x.tobytes(), 'big')
                             for x in pair], dtype=np.float64)
        lens = np.full(n, float(L))
        d: Dict[str, float] = {}
        # adjacent-column homogeneity
        d['cardchg'] = abs(math.log(l['card'] + 1) - math.log(r['card'] + 1))
        d['hchg'] = abs(l['H'] - r['H'])
        d['prchg'] = abs(l['printable'] - r['printable'])
        d['zchg'] = abs(l['zero'] - r['zero'])
        d['mi'] = mutual_information(M[:, j - 1], M[:, j])
        d['dep'] = 1.0 - cpair / max(1, l['card'] * r['card'])
        # multi-byte determinism
        d['pairdet'] = max(l['card'], r['card']) / max(1, cpair)
        d['lgain'] = math.log((cpair + 1) / (max(l['card'], r['card']) + 1))
        d['corrnum'] = abs(pearson(M[:, j - 1], M[:, j]))
        # single-column attributes
        d['maxcard'] = math.log(1 + max(l['card'], r['card'])) / math.log(1 + n)
        d['mincard'] = math.log(1 + min(l['card'], r['card'])) / math.log(1 + n)
        d['constl'] = float(l['const'])
        d['constr'] = float(r['const'])
        d['leftH'] = l['H']
        d['rightH'] = r['H']
        # positional prior
        d['relpos'] = j / L
        d['logpos'] = math.log(1 + j)
        d['even2'] = float(j % 2 == 0)
        d['even4'] = float(j % 4 == 0)
        d['is2tok'] = float(j in (2, 4, 8))
        # semantic region
        d['strl'] = l['printable']
        d['strr'] = r['printable']
        d['strboth'] = min(l['printable'], r['printable'])
        d['prun'] = float(np.minimum(run[:, j - 1], run[:, j]).mean()) / L
        d['prun_both'] = float((run[:, j - 1] + run[:, j]).mean()) / (2 * L)
        d['zrun'] = float(np.minimum(runz[:, j - 1], runz[:, j]).mean()) / L
        d['txt_rate'] = float(np.mean([
            (48 <= M[i, j - 1] <= 57 or 65 <= M[i, j - 1] <= 90
             or 97 <= M[i, j - 1] <= 122 or M[i, j - 1] == 32)
            and (48 <= M[i, j] <= 57 or 65 <= M[i, j] <= 90
                 or 97 <= M[i, j] <= 122 or M[i, j] == 32)
            for i in range(n)])) if n else 0.0
        # neighbour context
        d['prev_cardchg'] = (abs(math.log(lm1['card'] + 1)
                                 - math.log(l['card'] + 1))
                             if lm1 is not None else 0.0)
        d['prev_hchg'] = abs(lm1['H'] - l['H']) if lm1 is not None else 0.0
        d['prev_const'] = float(lm1['const']) if lm1 is not None else 1.0
        d['next_cardchg'] = (abs(math.log(r['card'] + 1)
                                 - math.log(rp1['card'] + 1))
                             if rp1 is not None else 0.0)
        d['next_hchg'] = abs(r['H'] - rp1['H']) if rp1 is not None else 0.0
        d['next_const'] = float(rp1['const']) if rp1 is not None else 1.0
        # scale and length correlation
        d['lencorr'] = abs(pearson(pair_int, lens))
        d['nn'] = math.log(1 + n)
        d['invL'] = math.log(1 + L)
        # length / sequence-field evidence
        rem = lens_arr - j
        if n:
            ok1 = (rem >= 0) & (rem <= 255)
            d['selflen1'] = float(
                np.mean((M[:, j - 1].astype(np.int64) == rem) & ok1))
            ok2 = (rem >= 0) & (rem <= 65535)
            if j >= 2:
                be = (M[:, j - 2].astype(np.int64) << 8) | M[:, j - 1]
                le = (M[:, j - 1].astype(np.int64) << 8) | M[:, j - 2]
                d['selflen2'] = max(float(np.mean((be == rem) & ok2)),
                                    float(np.mean((le == rem) & ok2)))
                d['totlen2'] = max(float(np.mean(be == lens_arr)),
                                   float(np.mean(le == lens_arr)))
            else:
                d['selflen2'] = 0.0
                d['totlen2'] = 0.0
            d['totlen1'] = float(
                np.mean(M[:, j - 1].astype(np.int64) == (lens_arr & 0xFF)))
            d['idxcorr'] = max(
                abs(pearson(M[:, j - 1].astype(np.float64), idx_arr)),
                abs(pearson(M[:, j].astype(np.float64), idx_arr)))
            d['cntinc'] = (float(np.mean(
                (M[1:, j - 1].astype(np.int32) - M[:-1, j - 1].astype(np.int32))
                == 1)) if n >= 3 else 0.0)
        else:
            d['selflen1'] = d['selflen2'] = 0.0
            d['totlen1'] = d['totlen2'] = 0.0
            d['idxcorr'] = d['cntinc'] = 0.0
        rows.append(d)
    return rows


def features_global(msgs: Sequence[bytes], L_true: int,
                    ctx: int = 2) -> List[Optional[Dict[str, float]]]:
    """Cross-stratum view: recompute each cut point from all messages that
    cover it, instead of from the current stratum only (Section 4.4.2)."""
    n_all = len(msgs)
    if n_all == 0:
        return []
    W = max(len(m) for m in msgs)
    G = np.zeros((n_all, W), dtype=np.uint8)
    for i, m in enumerate(msgs):
        G[i, :len(m)] = np.frombuffer(m, dtype=np.uint8)
    lens = np.array([len(m) for m in msgs], dtype=np.int64)
    out: List[Optional[Dict[str, float]]] = []
    for j in range(1, L_true):
        mask = lens > j
        if int(mask.sum()) < 2:
            out.append(None)
            continue
        lo = max(0, j - 1 - ctx)
        hi = min(W, j + 1 + ctx)
        sub = G[mask, lo:hi]
        w = sub.shape[1]
        rel = j - lo
        if rel < 1 or rel > w - 1:
            out.append(None)
            continue
        d = features_for_stratum(sub, w, row_len=lens[mask])[rel - 1]
        nn = int(mask.sum())
        d['relpos'] = j / L_true
        d['logpos'] = math.log(1 + j)
        d['even2'] = float(j % 2 == 0)
        d['even4'] = float(j % 4 == 0)
        d['is2tok'] = float(j in (2, 4, 8))
        d['nn'] = math.log(1 + nn)
        d['invL'] = math.log(1 + L_true)
        s = w / max(1, L_true)
        d['prun'] *= s
        d['prun_both'] *= s
        d['zrun'] *= s
        out.append(d)
    return out
