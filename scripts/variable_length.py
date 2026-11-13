"""Variable-length field identification (Table 5).

A reference field is *variable-length* when the same field (same name, same
start offset) occupies a different number of bytes in different messages of a
capture.  Boundaries are attributed per field rather than per offset:

* a reference boundary that starts a variable-length field and is predicted
  correctly counts as a true positive of the variable-length subtask, and as a
  false negative when it is missed;
* a predicted boundary that falls *inside* a variable-length field is a false
  positive of that subtask;
* everything else belongs to the fixed-length subtask.

The script also reports how many false positives that costs per 100 bytes of
variable-length payload, and (with ``--length-indicator``) whether a strongly
correlated length field sits in the eight bytes preceding a variable-length
field and whether its own start offset was recovered.

    python scripts/variable_length.py
    python scripts/variable_length.py --json outputs/varlen.json
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.common as C                    # noqa: E402
from m2rs.cli import load_config              # noqa: E402

CORR_THRESHOLD = 0.9      # |spearman| above which a length field is assumed


def _ranks(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind='mergesort')
    r = np.empty(len(a), dtype=np.float64)
    r[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return r


def spearman(x, y) -> float:
    """Rank correlation; robust to the outliers a length field can carry."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return 0.0
    with np.errstate(invalid='ignore', divide='ignore'):
        c = float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])
    return c if np.isfinite(c) else 0.0


def top_fields(rec):
    return [f for f in rec['fields'] if f.get('parent') is None]


def analyse_capture(records, length_indicator=False):
    """Variable/fixed-length subtask counts for one capture."""
    flen, starts = defaultdict(Counter), defaultdict(set)
    for rec in records:
        for f in top_fields(rec):
            s, w = int(f['start']), int(f['end']) - int(f['start'])
            flen[(f.get('name') or '', s)][w] += 1
            starts[f.get('name') or ''].add(s)
    var_keys = {k for k, c in flen.items() if len(c) > 1}
    drifting = {n for n, ss in starts.items() if len(ss) > 1}

    stat = {'var': Counter(), 'fix': Counter()}
    n_inst = span = n_drift = 0
    groups = defaultdict(lambda: {'lens': [], 'msgs': [], 'preds': []})

    for rec in records:
        preds = {int(p) for p in rec['cuts']}
        fields = [(int(f['start']), int(f['end']), f.get('name') or '')
                  for f in top_fields(rec)]
        is_var = {}
        for s, e, name in fields:
            if s <= 0:
                continue
            is_var[s] = is_var.get(s, False) or ((name, s) in var_keys)
        for s, flag in is_var.items():
            kind = 'var' if flag else 'fix'
            stat[kind]['tp' if s in preds else 'fn'] += 1
        for p in sorted(preds):
            inside = [(s, e, n) for (s, e, n) in fields if s < p < e]
            if not inside:
                continue
            s, e, n = max(inside, key=lambda t: (t[0], -t[1]))
            stat['var' if (n, s) in var_keys else 'fix']['fp'] += 1
        for s, e, name in fields:
            if (name, s) not in var_keys:
                continue
            n_inst += 1
            span += max(0, e - s)
            if name in drifting:
                n_drift += 1
            if s > 0 and e > s:
                g = groups[(name, s)]
                g['lens'].append(e - s)
                g['msgs'].append(rec['msg'])
                g['preds'].append(sorted(preds))

    out = {'n_var_inst': n_inst, 'var_span': span,
           'drift_rate': (n_drift / n_inst) if n_inst else 0.0,
           'n_var_names': len({n for (n, _s) in var_keys}),
           'var': dict(stat['var']), 'fix': dict(stat['fix'])}
    for kind in ('var', 'fix'):
        c = stat[kind]
        from m2rs.metrics import prf
        d = prf(c.get('tp', 0), c.get('fp', 0), c.get('fn', 0))
        out[kind + '_P'], out[kind + '_R'], out[kind + '_F1'] = (
            d['P'], d['R'], d['F1'])
    out['spurious_per_100b'] = (100.0 * stat['var'].get('fp', 0) / span
                                if span else 0.0)

    if length_indicator:
        n_grp = n_ok = n_obs = n_hit = 0
        corrs = []
        for (_name, s), g in groups.items():
            lens = g['lens']
            if len(lens) < 8 or len(set(lens)) < 2:
                continue
            n_grp += 1
            lens = np.asarray(lens, dtype=np.float64)
            best = (0.0, 0)
            for off in range(max(0, s - 8), s):
                for k in (1, 2, 4):
                    if off + k > s:
                        continue
                    seg = [m[off:off + k] for m in g['msgs']]
                    if any(len(x) != k for x in seg):
                        continue
                    for order in ('big', 'little'):
                        v = np.array([int.from_bytes(x, order) for x in seg],
                                     dtype=np.float64)
                        c = abs(spearman(v, lens))
                        if c > best[0]:
                            best = (c, off)
            corrs.append(best[0])
            if best[0] >= CORR_THRESHOLD:
                n_ok += 1
                if best[1] > 0:
                    for pl in g['preds']:
                        n_obs += 1
                        n_hit += 1 if best[1] in pl else 0
        out['length_indicator'] = {
            'groups': n_grp, 'detectable': n_ok,
            'detectable_rate': (n_ok / n_grp) if n_grp else 0.0,
            'instances': n_obs, 'identified': n_hit,
            'identified_rate': (n_hit / n_obs) if n_obs else 0.0,
            'mean_abs_corr': float(np.mean(corrs)) if corrs else 0.0,
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='variable_length', description=__doc__)
    ap.add_argument('--data', default=None)
    ap.add_argument('--tol', type=int, default=0)
    ap.add_argument('--length-indicator', action='store_true',
                    help='also test whether a correlated length field precedes '
                         'each variable-length field (slower)')
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--pool-w', type=float, default=None,
                    help='fusion weight w (default: config, 0.7)')
    ap.add_argument('--json', default='outputs/variable_length.json')
    a = ap.parse_args(argv)

    cfg = load_config()
    over = {} if a.pool_w is None else {'pool_w': a.pool_w}
    eng = C.make_engine(cfg, model_dir=a.model_dir, **over)
    ref = C.make_ref(cfg)

    per = {}
    tot = Counter()
    for pcap in C.capture_paths(a.data):
        proto = Path(pcap).stem
        recs = C.predict_capture(eng, pcap, cfg, ref)
        row = analyse_capture(recs, a.length_indicator)
        per[proto] = row
        for kind in ('var', 'fix'):
            for k in ('tp', 'fp', 'fn'):
                tot['%s_%s' % (kind, k)] += row[kind].get(k, 0)
        tot['span'] += row['var_span']
        tot['n_var'] += row['n_var_inst']
        print('%-10s var=%-5d P=%.3f R=%.3f F1=%.3f  fixed F1=%.3f  '
              'spurious/100B=%.2f'
              % (proto, row['n_var_inst'], row['var_P'], row['var_R'],
                 row['var_F1'], row['fix_F1'], row['spurious_per_100b']),
              flush=True)

    from m2rs.metrics import prf
    v = prf(tot['var_tp'], tot['var_fp'], tot['var_fn'])
    f = prf(tot['fix_tp'], tot['fix_fp'], tot['fix_fn'])
    summary = {'var_P': v['P'], 'var_R': v['R'], 'var_F1': v['F1'],
               'fix_P': f['P'], 'fix_R': f['R'], 'fix_F1': f['F1'],
               'var_tp': tot['var_tp'], 'var_fp': tot['var_fp'],
               'var_fn': tot['var_fn'], 'var_span': tot['span'],
               'n_var_inst': tot['n_var'],
               'spurious_per_100b': (100.0 * tot['var_fp'] / tot['span']
                                     if tot['span'] else 0.0)}
    print('-' * 84)
    print('%-10s var=%-5d P=%.3f R=%.3f F1=%.3f  fixed F1=%.3f  '
          'spurious/100B=%.2f'
          % ('overall', tot['n_var'], v['P'], v['R'], v['F1'], f['F1'],
             summary['spurious_per_100b']))

    C.write_json(a.json, {'tol': a.tol, 'summary': summary, 'per_protocol': per})


if __name__ == '__main__':
    main()
