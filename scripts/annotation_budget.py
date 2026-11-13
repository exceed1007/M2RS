"""Accuracy as a function of the annotated corpus size (Table 6).

The discriminator is trained on labelled strata, and a stratum (one message
template at one message length) is the smallest unit that can be annotated:
the labels inside a stratum are strongly correlated, so sampling individual
positions would overstate what a small budget buys.  This script therefore
samples *whole strata*, refits, and evaluates on the test captures.

    python scripts/annotation_budget.py                       # 6 budgets, 3 draws
    python scripts/annotation_budget.py --ratios 0.01,0.1,1.0 --seeds 7

The width prior is held fixed at ``models/mmfi/len_prior.json`` so that only
the amount of annotation behind the discriminator varies and the 100 % point
coincides with the model used for Table 1 (``--no-fix-prior`` recomputes it
from the sampled strata instead).
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.common as C                                # noqa: E402
from m2rs.cli import load_config                          # noqa: E402
from m2rs.train import (build_corpus, collect_files,      # noqa: E402
                        file_md5)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='annotation_budget', description=__doc__)
    ap.add_argument('--ratios', default='0.01,0.05,0.10,0.25,0.50,1.0')
    ap.add_argument('--seeds', default='7,17,27',
                    help='repeated draws, one row per seed')
    ap.add_argument('--known-dir', default=None)
    ap.add_argument('--data', default=None)
    ap.add_argument('--max-files', type=int, default=None)
    ap.add_argument('--max-messages', type=int, default=None)
    ap.add_argument('--min-stratum', type=int, default=None)
    ap.add_argument('--no-fix-prior', dest='fix_prior', action='store_false',
                    help='recompute the width prior from the sampled strata')
    ap.add_argument('--tol', type=int, default=0)
    ap.add_argument('--json', default='outputs/annotation_budget.json')
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
    files = collect_files(known, exclude, max_files, 7)
    print('captures: %d selected from %s' % (len(files), known), flush=True)

    X, y, names, groups, _mid, _last = build_corpus(
        files, cfg, min_stratum, max_messages)
    if len(X) == 0:
        raise SystemExit('no training sample collected')
    groups = np.asarray(groups)
    print('corpus: %d positions / %d strata, positive rate %.3f'
          % (len(X), len(set(groups)), y.mean()), flush=True)

    fixed = None
    if a.fix_prior:
        fixed = json.loads((ROOT / 'models' / 'mmfi' /
                            'len_prior.json').read_text(encoding='utf-8'))

    ref = C.make_ref(cfg)
    tmp = Path(tempfile.mkdtemp(prefix='m2rs_budget_'))
    rows = []
    try:
        for ratio in C.csv_numbers(a.ratios):
            for seed in [int(s) for s in str(a.seeds).split(',') if s.strip()]:
                t0 = time.time()
                picked = set(C.pick_groups(groups, ratio, seed))
                mask = np.array([g in picked for g in groups])
                C.refit(X[mask], y[mask], names, groups[mask], tmp,
                        seed=seed, prior=fixed)
                eng = C.make_engine(cfg, model_dir=str(tmp))
                res = C.evaluate(eng, cfg, a.data, a.tol, ref=ref)
                rows.append({'ratio': ratio, 'seed': seed,
                             'n_strata': len(picked),
                             'n_positions': int(mask.sum()),
                             'P': res['micro']['P'], 'R': res['micro']['R'],
                             'F1': res['micro']['F1'],
                             'seconds': round(time.time() - t0, 1)})
                print('r=%-5.2f seed=%-3d strata=%-5d positions=%-7d '
                      'F1=%.4f (P=%.4f R=%.4f)  %ds'
                      % (ratio, seed, len(picked), mask.sum(),
                         res['micro']['F1'], res['micro']['P'],
                         res['micro']['R'], rows[-1]['seconds']), flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    by = {}
    for r in rows:
        by.setdefault(r['ratio'], []).append(r['F1'])
    summary = []
    for ratio in sorted(by):
        v = np.asarray(by[ratio])
        sub = [r for r in rows if r['ratio'] == ratio]
        summary.append({'ratio': ratio,
                        'n_strata': int(np.mean([s['n_strata'] for s in sub])),
                        'n_positions': int(np.mean([s['n_positions']
                                                    for s in sub])),
                        'P': float(np.mean([s['P'] for s in sub])),
                        'R': float(np.mean([s['R'] for s in sub])),
                        'F1_mean': float(v.mean()),
                        'F1_std': float(v.std()),
                        'draws': int(v.size)})
    print('\n%-8s %8s %10s %8s %8s %14s'
          % ('budget', 'strata', 'positions', 'P', 'R', 'F1'))
    for s in summary:
        print('%-8.0f%% %8d %10d %8.3f %8.3f   %.3f +- %.3f'
              % (100 * s['ratio'], s['n_strata'], s['n_positions'], s['P'],
                 s['R'], s['F1_mean'], s['F1_std']))

    C.write_json(a.json, {'tol': a.tol, 'fix_prior': a.fix_prior,
                          'summary': summary, 'rows': rows})


if __name__ == '__main__':
    main()
