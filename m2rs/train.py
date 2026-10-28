"""Train the general boundary discriminator and the field-width prior.

Corpus unit: one *length stratum* (a protocol capture at one message length)
whose messages share a single reference template. Each interior offset j of
such a stratum is a sample:

    x_j : position-level features of j
    y_j : 1 if the reference template cuts at j

Outputs (models/mmfi/): boundary_scorer.joblib, boundary_scorer_lr.json,
len_prior.json.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .cli import ROOT, load_config
from .dataset import TsharkReference, load_sessions
from .features import features_for_stratum

CAPTURE_SUFFIXES = ('*.pcap', '*.pcapng', '*.cap')


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def collect_files(known_dir: Path, exclude: set, limit: int,
                  seed: int = 7) -> List[Path]:
    """Pick up to ``limit`` captures, round-robin over source folders."""
    files = []
    for suf in CAPTURE_SUFFIXES:
        files.extend(known_dir.rglob(suf))
    files = sorted(set(files))

    keep = []
    for f in files:
        try:
            if file_md5(f) in exclude or f.stat().st_size < 200:
                continue
        except OSError:
            continue
        keep.append(f)

    random.Random(seed).shuffle(keep)
    by_top: Dict[str, List[Path]] = {}
    for f in keep:
        rel = f.relative_to(known_dir)
        by_top.setdefault(rel.parts[0] if len(rel.parts) > 1 else '.', []).append(f)

    picked: List[Path] = []
    tops = sorted(by_top)
    i = 0
    while len(picked) < limit:
        added = False
        for t in tops:
            if i < len(by_top[t]):
                picked.append(by_top[t][i])
                added = True
                if len(picked) >= limit:
                    break
        if not added:
            break
        i += 1
    return picked


def stratum_samples(messages: Sequence[bytes],
                    templates: Sequence[Sequence[int]],
                    min_stratum: int) -> List[Tuple[int, List[bytes], List[int]]]:
    """Group messages into strata that share one template."""
    groups: Dict[Tuple[int, Tuple[int, ...]], List[bytes]] = {}
    for msg, tmpl in zip(messages, templates):
        if len(msg) < 3:
            continue
        groups.setdefault((len(msg), tuple(sorted(tmpl))), []).append(msg)
    out = []
    for (length, tmpl), msgs in groups.items():
        if len(msgs) >= min_stratum and tmpl:
            out.append((length, msgs, list(tmpl)))
    return out


def build_corpus(files: Sequence[Path], cfg: dict, min_stratum: int,
                 max_messages: int, first_level: bool = True
                 ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Counter, Counter]:
    """Return X, y, feature names, groups and width counters."""
    ref = TsharkReference(cfg['dataset'].get('tshark', 'tshark'),
                          int(cfg['dataset'].get('tshark_timeout', 300)))
    thr = float(cfg['dataset'].get('ascii_threshold', 0.4))
    X, y, groups = [], [], []
    mid, last = Counter(), Counter()

    for pcap in files:
        try:
            frame_fields, payload_info = ref.load(str(pcap))
            sessions = load_sessions(str(pcap), thr)
        except Exception as e:                      # unreadable capture
            print('[skip] %-40s %s' % (pcap.name, type(e).__name__))
            continue

        msgs, tmpls = [], []
        for sess in sessions:
            for i, msg in enumerate(sess.messages):
                fields = ref.fields_for_frames(sess.frames[i], frame_fields,
                                               payload_info, len(msg))
                if not fields:
                    continue
                msgs.append(msg)
                tmpls.append(ref.positions(fields, len(msg), first_level))
                if len(msgs) >= max_messages:
                    break
            if len(msgs) >= max_messages:
                break
        if len(msgs) < 4:
            continue

        n_strata = 0
        for length, members, tmpl in stratum_samples(msgs, tmpls, min_stratum):
            M = np.zeros((len(members), length), dtype=np.uint8)
            for r, m in enumerate(members):
                M[r] = np.frombuffer(m, dtype=np.uint8)
            cut = set(tmpl)
            for j1, row in enumerate(features_for_stratum(M, length)):
                X.append([row[k] for k in sorted(row)])
                y.append(1 if (j1 + 1) in cut else 0)
                groups.append('%s#%d' % (pcap.stem, length))
            edges = [0] + sorted(cut) + [length]
            for k in range(len(edges) - 1):
                (last if k == len(edges) - 2 else mid)[edges[k + 1] - edges[k]] += 1
            n_strata += 1
        print('[ok]   %-40s messages=%-5d strata=%d' % (pcap.name, len(msgs), n_strata))

    names = sorted(features_for_stratum(np.zeros((3, 4), dtype=np.uint8), 4)[0])
    return (np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int8),
            names, groups, mid, last)


def width_prior(mid: Counter, last: Counter, max_w: int = 64,
                eps: float = 1e-5) -> dict:
    def norm(cnt):
        tot = sum(cnt.values()) or 1
        p = {str(w): max(cnt.get(w, 0) / tot, eps) for w in range(1, max_w + 1)}
        s = sum(p.values())
        return {k: v / s for k, v in p.items()}

    return {'max_len': max_w, 'p_mid': norm(mid), 'p_last': norm(last),
            'tail_rate_mid': 0.35, 'tail_rate_last': 0.02}


def fit(X: np.ndarray, y: np.ndarray, names: List[str], groups: List[str],
        out_dir: Path, seed: int = 7, with_gb: bool = True) -> dict:
    """Fit the logistic-regression and gradient-boosting discriminators."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    ug = sorted(set(groups))
    rng = np.random.RandomState(seed)
    rng.shuffle(ug)
    val = set(ug[:max(1, int(len(ug) * 0.25))])
    is_val = np.array([g in val for g in groups])
    Xtr, ytr = X[~is_val], y[~is_val]
    Xva, yva = X[is_val], y[is_val]

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-9] = 1.0

    def std(Z):
        return (Z - mu) / sd

    report = {}
    lr = LogisticRegression(max_iter=4000, C=1.0, class_weight='balanced')
    lr.fit(std(Xtr), ytr)
    for tag, Xs, ys in (('train', Xtr, ytr), ('val', Xva, yva)):
        p = lr.predict_proba(std(Xs))[:, 1]
        report['lr_' + tag] = {'auc': float(roc_auc_score(ys, p)),
                               'ap': float(average_precision_score(ys, p))}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'boundary_scorer_lr.json').write_text(json.dumps({
        'names': names, 'mu': mu.tolist(), 'sd': sd.tolist(),
        'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0]),
    }, indent=1), encoding='utf-8')

    if with_gb:
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier
        gb = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=seed)
        gb.fit(std(Xtr), ytr)
        for tag, Xs, ys in (('train', Xtr, ytr), ('val', Xva, yva)):
            p = gb.predict_proba(std(Xs))[:, 1]
            report['gb_' + tag] = {'auc': float(roc_auc_score(ys, p)),
                                   'ap': float(average_precision_score(ys, p))}
        joblib.dump({'model': gb, 'names': names, 'mu': mu, 'sd': sd},
                    out_dir / 'boundary_scorer.joblib')
    return report


def main(argv=None):
    cfg_all = load_config()
    tcfg = cfg_all.get('training') or {}
    ap = argparse.ArgumentParser(prog='m2rs.train', description=__doc__)
    ap.add_argument('--known-dir', default=str(ROOT / tcfg.get(
        'known_dir', './data/known_protocol')))
    ap.add_argument('--out-dir', default=str(ROOT / tcfg.get(
        'out_dir', './models/mmfi')))
    ap.add_argument('--max-files', type=int, default=int(tcfg.get('max_files', 200)))
    ap.add_argument('--max-messages', type=int,
                    default=int(tcfg.get('max_messages', 250)))
    ap.add_argument('--min-stratum', type=int,
                    default=int(tcfg.get('min_stratum', 3)))
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--no-gb', action='store_true')
    a = ap.parse_args(argv)

    known = Path(a.known_dir)
    if not known.exists():
        raise SystemExit('known-protocol directory not found: %s' % known)

    # Captures used for testing must never leak into training.
    exclude = set()
    for p in (ROOT / 'data' / 'test_protocol').glob('*'):
        if p.is_file():
            exclude.add(file_md5(p))

    files = collect_files(known, exclude, a.max_files, a.seed)
    print('captures: %d selected from %s' % (len(files), known))
    X, y, names, groups, mid, last = build_corpus(
        files, cfg_all, a.min_stratum, a.max_messages)
    if len(X) == 0:
        raise SystemExit('no training sample collected')
    print('samples=%d positives=%.3f features=%d strata=%d'
          % (len(X), y.mean(), len(names), len(set(groups))))

    out = Path(a.out_dir)
    report = fit(X, y, names, groups, out, a.seed, with_gb=not a.no_gb)
    prior = width_prior(mid, last, int(tcfg.get('max_field_len', 64)))
    (out / 'len_prior.json').write_text(json.dumps(prior, indent=1),
                                        encoding='utf-8')
    print(json.dumps(report, indent=1))
    print('written to %s' % out)


if __name__ == '__main__':
    main()
