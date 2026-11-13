"""Leave-one-source-out validation of the universal feature set (Table 2).

The feature set is pooled from several *sources* (the top-level folders of the
known-protocol corpus, roughly one per collection or protocol family).  To show
that it does not depend on any single one of them, each fold removes one source,
refits the discriminator and the width prior on the rest, and then measures

* the held-out source itself, whose captures never entered training, and
* the twelve test protocols, to see how far the overall result moves.

    python scripts/leave_one_out.py --folds 7
    python scripts/leave_one_out.py --folds 7 --only ICS-Pcaps-master

The corpus is built once; a fold only refits on the subset, so the whole run
costs one pass of tshark over the known captures.
"""

import argparse
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.common as C                                # noqa: E402
from m2rs.cli import load_config                          # noqa: E402
from m2rs.metrics import match_positions, prf             # noqa: E402
from m2rs.train import (build_corpus, collect_files,      # noqa: E402
                        file_md5)


def held_out_perf(engine, blobs, keys):
    """Score the removed source on its own strata (one template per stratum)."""
    acc = [0, 0, 0]
    for k in keys:
        M, tmpl = blobs[k]
        if not tmpl:
            continue
        msgs = [bytes(M[r]) for r in range(M.shape[0])]
        for pred in engine.predict_messages(msgs):
            t = match_positions(pred, list(tmpl), 0)
            acc = [acc[i] + t[i] for i in range(3)]
    if acc == [0, 0, 0]:
        return None
    d = prf(*acc)
    return {'P': d['P'], 'R': d['R'], 'F1': d['F1'], 'tp': acc[0],
            'fp': acc[1], 'fn': acc[2]}


def main(argv=None):
    ap = argparse.ArgumentParser(prog='leave_one_out', description=__doc__)
    ap.add_argument('--known-dir', default=None,
                    help='captures of known protocols (default: config)')
    ap.add_argument('--data', default=None,
                    help='test captures (default: data/test_protocol)')
    ap.add_argument('--folds', type=int, default=7,
                    help='number of sources to hold out, largest first')
    ap.add_argument('--only', default=None,
                    help='comma-separated source names, prefix match')
    ap.add_argument('--max-files', type=int, default=None)
    ap.add_argument('--max-messages', type=int, default=None)
    ap.add_argument('--min-stratum', type=int, default=None)
    ap.add_argument('--tol', type=int, default=0)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--json', default='outputs/leave_one_out.json')
    a = ap.parse_args(argv)

    cfg = load_config()
    tcfg = cfg.get('training') or {}
    known = Path(a.known_dir or ROOT / tcfg.get('known_dir',
                                                './data/known_protocol'))
    if not known.exists():
        raise SystemExit('known-protocol directory not found: %s\n'
                         'See data/README.md for where to obtain it.' % known)
    max_files = a.max_files or int(tcfg.get('max_files', 200))
    max_messages = a.max_messages or int(tcfg.get('max_messages', 250))
    min_stratum = a.min_stratum or int(tcfg.get('min_stratum', 3))

    exclude = {file_md5(p) for p in
               (ROOT / 'data' / 'test_protocol').glob('*') if p.is_file()}
    files = collect_files(known, exclude, max_files, a.seed)
    print('captures: %d selected from %s' % (len(files), known), flush=True)

    blobs, origins = {}, {}
    X, y, names, groups, _mid, _last = build_corpus(
        files, cfg, min_stratum, max_messages, blobs=blobs, origins=origins)
    if len(X) == 0:
        raise SystemExit('no training sample collected')

    def source_of(group):
        p = Path(origins[group]).relative_to(known)
        return p.parts[0] if len(p.parts) > 1 else '.'

    gtop = np.array([source_of(g) for g in groups])
    cnt = Counter(gtop.tolist())
    print('samples=%d strata=%d sources=%d'
          % (len(X), len(blobs), len(cnt)), flush=True)
    for k, v in cnt.most_common():
        print('   %-40s %8d samples' % (k, v))

    folds = [k for k, _ in cnt.most_common(a.folds)]
    if a.only:
        want = [s.strip() for s in a.only.split(',') if s.strip()]
        exact = [f for f in folds if f in want]
        folds = exact or [f for f in folds if any(w in f for w in want)]

    ref = C.make_ref(cfg)
    tmp = Path(tempfile.mkdtemp(prefix='m2rs_loo_'))
    rows = []
    try:
        for src in folds:
            t0 = time.time()
            keep = gtop != src
            C.refit(X[keep], y[keep], names,
                    [g for g, k in zip(groups, keep) if k], tmp, a.seed)
            eng = C.make_engine(cfg, model_dir=str(tmp))
            test = C.evaluate(eng, cfg, a.data, a.tol, ref=ref)
            keys = [g for g in blobs if source_of(g) == src]
            held = held_out_perf(eng, blobs, keys) if len(keys) >= 3 else None
            rows.append({'held_out': src, 'n_samples': int(cnt[src]),
                         'n_strata': len(keys), 'held_out_perf': held,
                         'test': {'P': test['micro']['P'],
                                  'R': test['micro']['R'],
                                  'F1': test['micro']['F1']},
                         'test_per_protocol': {
                             k: {'F1': v['F1']}
                             for k, v in test['per_protocol'].items()},
                         'seconds': round(time.time() - t0, 1)})
            hs = ('P=%.3f R=%.3f F1=%.3f' % (held['P'], held['R'], held['F1'])
                  if held else 'n/a')
            print('hold out %-34s strata=%-4d held-out %s   test F1=%.4f  (%ds)'
                  % (src, len(keys), hs, test['micro']['F1'],
                     rows[-1]['seconds']), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    f1s = [r['test']['F1'] for r in rows]
    hf1 = [r['held_out_perf']['F1'] for r in rows if r['held_out_perf']]
    full = C.evaluate(C.make_engine(cfg), cfg, a.data, a.tol, ref=ref)
    summary = {
        'full_test_F1': full['micro']['F1'],
        'loo_test_F1_mean': float(np.mean(f1s)),
        'loo_test_F1_std': float(np.std(f1s)),
        'loo_test_F1_min': float(np.min(f1s)),
        'loo_test_F1_max': float(np.max(f1s)),
        'held_out_F1_mean': float(np.mean(hf1)) if hf1 else None,
        'held_out_F1_min': float(np.min(hf1)) if hf1 else None,
        'held_out_F1_max': float(np.max(hf1)) if hf1 else None,
    }
    print('\nfull model test F1 = %.4f' % summary['full_test_F1'])
    print('leave-one-out test F1: mean %.4f  s.d. %.4f  range %.4f-%.4f'
          % (summary['loo_test_F1_mean'], summary['loo_test_F1_std'],
             summary['loo_test_F1_min'], summary['loo_test_F1_max']))
    if hf1:
        print('held-out source F1: mean %.4f  min %.4f  max %.4f'
              % (summary['held_out_F1_mean'], summary['held_out_F1_min'],
                 summary['held_out_F1_max']))

    C.write_json(a.json, {'tol': a.tol, 'max_files': max_files,
                          'summary': summary, 'folds': rows})


if __name__ == '__main__':
    main()
