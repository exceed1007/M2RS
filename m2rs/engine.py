"""M2RS field-boundary inference engine (Section 4).

Pipeline: length stratification (4.2) -> position-level scoring with the
general feature set (4.3) -> width prior and multi-scale fusion (4.4) ->
MAP decoding with consistency check (4.5).
"""

import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .features import FEATURE_FAMILIES, features_for_stratum, features_global

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'mmfi')
NEG = -1e18

# Parameters reported in Section 5.1.
DEFAULT_CFG = {
    'theta': -2.8,
    'min_support': 2,
    'pool_max_n': 4,
    'pool_w': 0.7,
    'widen_prior': 1.0,
    'max_len': 0,
    'kind': 'gb',
    # Ablation switches; all enabled means the full method.
    'use_stratify': True,
    'use_scorer': True,
    'use_prior': True,
    'use_fusion': True,
    'use_map': True,
    'drop_features': '',
}


class BoundaryScorer:
    """Position-level boundary discriminator.

    Loads the gradient-boosting model when available and falls back to the
    logistic-regression parameters shipped as JSON, so the repository runs
    without scikit-learn installed.
    """

    def __init__(self, kind: str = 'gb', model_dir: str = None,
                 drop: Sequence[str] = ()):
        self.kind = kind
        self.dir = model_dir or MODEL_DIR
        self.gb = None
        self.names = None
        self.mu = self.sd = self.w = None
        self.b = 0.0
        self.ready = False
        self.drop_idx: List[int] = []
        self._drop = set(drop or ())
        if kind == 'gb':
            self._load_gb()
        if not self.ready:
            self._load_lr()
        if self.names is not None and self._drop:
            self.drop_idx = [i for i, n in enumerate(self.names)
                             if n in self._drop]

    def _load_gb(self):
        path = os.path.join(self.dir, 'boundary_scorer.joblib')
        try:
            import joblib
            d = joblib.load(path)
            self.gb = d['model']
            self.names = list(d['names'])
            self.mu = np.asarray(d['mu'], dtype=np.float64)
            self.sd = np.asarray(d['sd'], dtype=np.float64)
            self.ready = True
        except Exception:
            self.gb = None

    def _load_lr(self):
        path = os.path.join(self.dir, 'boundary_scorer_lr.json')
        try:
            with open(path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            self.names = list(d['names'])
            self.mu = np.array(d['mu'], dtype=np.float64)
            self.sd = np.array(d['sd'], dtype=np.float64)
            self.w = np.array(d['coef'], dtype=np.float64)
            self.b = float(d.get('intercept', 0.0))
            self.ready = True
        except Exception:
            self.ready = False

    def _vector(self, rows) -> np.ndarray:
        X = np.array([[r[k] for k in self.names] for r in rows],
                     dtype=np.float64)
        if self.drop_idx:
            for i in self.drop_idx:
                X[:, i] = self.mu[i]
        return X

    def prob(self, rows) -> np.ndarray:
        if not self.ready:
            # Degraded mode: positional prior only.
            return np.array([max(0.02, min(0.9, 0.6 - r.get('relpos', 0)))
                             for r in rows], dtype=np.float64)
        X = self._vector(rows)
        Z = (X - self.mu) / self.sd
        if self.gb is not None:
            return self.gb.predict_proba(Z)[:, 1]
        z = Z @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class WidthPrior:
    """Field-width prior P_mid(w) / P_last(w) (Section 4.4.1)."""

    def __init__(self, path: str = None):
        self.path = path or os.path.join(MODEL_DIR, 'len_prior.json')
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                d = json.load(f)
            self.maxw = int(d['max_len'])
            self.p_mid = {int(k): float(v) for k, v in d['p_mid'].items()}
            self.p_last = {int(k): float(v) for k, v in d['p_last'].items()}
            self.tr_mid = float(d.get('tail_rate_mid', 0.35))
            self.tr_last = float(d.get('tail_rate_last', 0.02))
        except Exception:
            self.maxw = 8
            self.p_mid = {1: .548, 2: .327, 3: .012, 4: .044, 5: .012,
                          6: .006, 7: .005, 8: .008}
            self.p_last = {1: .164, 2: .207, 3: .030, 4: .072, 5: .030,
                           6: .030, 7: .020, 8: .034}
            self.tr_mid, self.tr_last = 0.35, 0.02
        self._c: Dict[tuple, float] = {}

    def phi(self, w: int, last: bool = False) -> float:
        key = (w, last)
        v = self._c.get(key)
        if v is not None:
            return v
        table = self.p_last if last else self.p_mid
        rate = self.tr_last if last else self.tr_mid
        p = table.get(w)
        if p is not None:
            v = math.log(max(p, 1e-9))
        else:
            v = math.log(max(min(table.values()), 1e-9)) - rate * (w - self.maxw)
        self._c[key] = v
        return v


def decode_map(L: int, logits: Sequence[float], prior: WidthPrior,
               theta: float = 0.0, max_len: int = 0,
               widen_prior: float = 1.0) -> List[int]:
    """Maximise sum(l_t - theta) + sum(log P(w)) over cut sets (Section 4.5.1).

    Cuts are the only decision variables; the last block uses the
    end-of-message width prior.
    """
    if L <= 1:
        return []
    s = np.array([float(x) - theta for x in logits], dtype=np.float64)
    limit = max_len if max_len and max_len > 0 else L
    dp = np.full(L + 1, NEG, dtype=np.float64)
    bk = np.zeros(L + 1, dtype=np.int64)
    dp[0] = 0.0
    for j in range(1, L + 1):
        best, bs = NEG, 0
        for a in range(max(0, j - limit), j):
            if dp[a] == NEG:
                continue
            v = dp[a] + widen_prior * prior.phi(j - a, last=False)
            if a > 0:
                v += s[a - 1]
            if v > best:
                best, bs = v, a
        dp[j] = best
        bk[j] = bs
    best, blast = NEG, 0
    for a in range(max(0, L - limit), L):
        if dp[a] == NEG:
            continue
        v = dp[a] + widen_prior * prior.phi(L - a, last=True)
        if a > 0:
            v += s[a - 1]
        if v > best:
            best, blast = v, a
    if best == NEG:
        return []
    out = []
    j = blast
    while j > 0:
        a = int(bk[j])
        if a > 0:
            out.append(a)
        j = a
    return sorted(out)


def decode_greedy(L: int, logits: Sequence[float], prior: WidthPrior,
                  theta: float = 0.0, max_len: int = 0,
                  widen_prior: float = 1.0) -> List[int]:
    """Local greedy decoding, used only by the MAP ablation."""
    if L <= 1:
        return []
    s = np.array([float(x) - theta for x in logits], dtype=np.float64)
    limit = max_len if max_len and max_len > 0 else L
    cuts: List[int] = []
    last = 0
    for j in range(1, L):
        w = j - last
        if limit and w + 1 > limit:
            cuts.append(j)
            last = j
            continue
        gain = s[j - 1] + widen_prior * (prior.phi(w) + prior.phi(1)
                                         - prior.phi(w + 1))
        if gain > 0:
            cuts.append(j)
            last = j
    return cuts


def homogeneity_logits(rows: Sequence[Dict[str, float]],
                       ref: Sequence[float] = None) -> np.ndarray:
    """Unsupervised stand-in for the discriminator in the scoring ablation.

    When ``ref`` is given the scores are linearly aligned to its mean and
    standard deviation, so both scoring paths share one decision scale and
    theta needs no recalibration.
    """
    if not rows:
        return np.zeros(0, dtype=np.float64)
    h = np.array([
        r.get('cardchg', 0.0) + r.get('hchg', 0.0)
        + 0.5 * r.get('prchg', 0.0) + 0.5 * r.get('zchg', 0.0)
        - 0.5 * r.get('strboth', 0.0)
        for r in rows], dtype=np.float64)
    sd = float(h.std())
    if sd < 1e-9:
        return np.full(len(h), float(np.mean(ref)) if len(ref) else 0.0)
    z = (h - float(h.mean())) / sd
    if ref is None or len(ref) == 0:
        return z
    r = np.asarray(ref, dtype=np.float64)
    rsd = float(r.std())
    return z * (rsd if rsd > 1e-9 else 1.0) + float(r.mean())


def _logit(p) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


class M2RSBoundaryEngine:
    """End-to-end boundary inference for one session."""

    def __init__(self, config: Optional[Dict] = None, model_dir: str = None):
        cfg = dict(DEFAULT_CFG)
        mcfg = (((config or {}).get('boundary') or {}).get('m2rs') or {})
        cfg.update({k: v for k, v in mcfg.items() if k in DEFAULT_CFG})
        self.cfg = cfg
        drop = [x.strip() for x in str(cfg.get('drop_features') or '').split(',')
                if x.strip()]
        fam = [FEATURE_FAMILIES[x] for x in drop if x in FEATURE_FAMILIES]
        if fam:
            drop = [x for x in drop if x not in FEATURE_FAMILIES] + \
                [n for f in fam for n in f]
        self.drop_features = drop
        self.scorer = BoundaryScorer(cfg.get('kind', 'gb'), model_dir,
                                     drop=self.drop_features)
        self.prior = WidthPrior(
            os.path.join(model_dir or MODEL_DIR, 'len_prior.json'))
        self.max_len = int(cfg.get('max_len', 0))
        self.timing: Dict[str, float] = {}

    def reset_timing(self):
        self.timing = {}

    def _tick(self, key: str, t0: float):
        self.timing[key] = self.timing.get(key, 0.0) + (time.time() - t0)

    def infer_matrix(self, M: np.ndarray, L: int,
                     all_msgs: Sequence[bytes] = None,
                     n_local: int = 0) -> List[int]:
        """Infer the template of one stratum given its n x L byte matrix."""
        t0 = time.time()
        rows = features_for_stratum(M, L)
        self._tick('features', t0)
        t0 = time.time()
        logs = _logit(self.scorer.prob(rows))
        if not self.cfg.get('use_scorer', True):
            logs = homogeneity_logits(rows, ref=logs)
        self._tick('scoring', t0)
        w = float(self.cfg.get('pool_w', 0.7))
        if (self.cfg.get('use_fusion', True) and all_msgs is not None
                and n_local <= int(self.cfg.get('pool_max_n', 4))
                and w > 0):
            t0 = time.time()
            grows = features_global(all_msgs, L)
            sel = [i for i, row in enumerate(grows)
                   if i < len(logs) and row is not None]
            if sel:
                if self.cfg.get('use_scorer', True):
                    vals = _logit(self.scorer.prob([grows[i] for i in sel]))
                else:
                    vals = np.array([homogeneity_logits([grows[i]],
                                                        ref=logs[i:i + 1])[0]
                                     for i in sel], dtype=np.float64)
                lg = logs.copy()
                for i, v in zip(sel, vals):
                    lg[i] = float(v)
                logs = (1 - w) * logs + w * lg
            self._tick('fusion', t0)
        t0 = time.time()
        theta = float(self.cfg.get('theta', -2.8))
        widen = (float(self.cfg.get('widen_prior', 1.0))
                 if self.cfg.get('use_prior', True) else 0.0)
        if self.cfg.get('use_map', True):
            T = decode_map(L, logs, self.prior, theta=theta,
                           max_len=self.max_len, widen_prior=widen)
        else:
            T = decode_greedy(L, logs, self.prior, theta=theta,
                              max_len=self.max_len, widen_prior=widen)
        self._tick('decoding', t0)
        return [b for b in T if 0 < b < L]

    def infer_stratum(self, msgs: Sequence[bytes], idxs: Sequence[int],
                      all_msgs: Sequence[bytes], L: int) -> List[int]:
        sub = [msgs[i] for i in idxs]
        M = np.zeros((len(sub), L), dtype=np.uint8)
        for r, m in enumerate(sub):
            M[r, :len(m)] = np.frombuffer(m, dtype=np.uint8)
        return self.infer_matrix(M, L, all_msgs, n_local=len(idxs))

    def predict_messages(self, msgs: Sequence[bytes]) -> List[List[int]]:
        """Boundary positions for every message of a session."""
        preds: List[List[int]] = [[] for _ in msgs]
        if not msgs:
            return preds
        if not self.cfg.get('use_stratify', True):
            # Stratification ablation: one zero-padded matrix for all messages.
            t0 = time.time()
            L = max(len(m) for m in msgs)
            if L < 3:
                return preds
            M = np.zeros((len(msgs), L), dtype=np.uint8)
            for r, m in enumerate(msgs):
                M[r, :len(m)] = np.frombuffer(m, dtype=np.uint8)
            T = self.infer_matrix(M, L, msgs, n_local=len(msgs))
            self._tick('stratify', t0)
            for i, m in enumerate(msgs):
                preds[i] = [b for b in T if 0 < b < len(m)]
            return preds
        t0 = time.time()
        strata: Dict[int, List[int]] = {}
        for i, m in enumerate(msgs):
            strata.setdefault(len(m), []).append(i)
        self._tick('stratify', t0)
        min_support = int(self.cfg.get('min_support', 2))
        templates: Dict[int, List[int]] = {}
        supported: List[tuple] = []
        for L in sorted(strata):
            idxs = strata[L]
            if len(idxs) < min_support or L < 3:
                templates[L] = []
                continue
            templates[L] = self.infer_stratum(msgs, idxs, msgs, L)
            supported.append((L, templates[L], len(idxs)))
        # Low-support strata reuse the template of the closest larger stratum.
        for L in sorted(strata):
            if templates.get(L):
                continue
            cands = [(Lp, Tp) for (Lp, Tp, _) in supported if Lp >= L]
            if not cands:
                cands = [(Lp, Tp) for (Lp, Tp, _) in supported]
            if not cands:
                continue
            Lp, Tp = max(cands, key=lambda x: x[0])
            templates[L] = [b for b in Tp if 0 < b < L]
        for L, idxs in strata.items():
            T = templates.get(L) or []
            for i in idxs:
                preds[i] = list(T)
        return preds

    def predict_global(self, msgs: Sequence[bytes]) -> List[int]:
        """Session-level boundary set used for visualisation."""
        preds = self.predict_messages(msgs)
        if not preds:
            return []
        cnt: Dict[int, int] = {}
        for p in preds:
            for b in p:
                cnt[b] = cnt.get(b, 0) + 1
        keep = [b for b, c in cnt.items() if c >= max(1, len(preds) // 3)]
        return sorted(keep)
