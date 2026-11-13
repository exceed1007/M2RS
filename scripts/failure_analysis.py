"""Where the residual errors sit (Table 7).

Recall is decomposed along four properties that can be read off the reference
annotation, so that a failure can be described rather than merely counted:

* the width of the field the boundary opens,
* whether that field is a bit field,
* whether the boundary opens a first-level or a nested field,
* the length of the message, and the protocol family.

False positives are characterised separately, by the distance to the nearest
reference boundary: a mismatch one or two bytes away is a near miss at a wide
field, a mismatch five bytes away is an invented cut.

    python scripts/failure_analysis.py
    python scripts/failure_analysis.py --json outputs/failure_modes.json
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.common as C                    # noqa: E402
from m2rs.cli import load_config              # noqa: E402


def width_bucket(w):
    if w is None:
        return 'n/a'
    if w <= 1:
        return '1 B'
    if w == 2:
        return '2 B'
    if w <= 4:
        return '3-4 B'
    if w <= 8:
        return '5-8 B'
    return '>8 B'


def length_bucket(n):
    if n <= 16:
        return '<=16 B'
    if n <= 32:
        return '17-32 B'
    if n <= 64:
        return '33-64 B'
    return '>64 B'


def field_at(fields, pos):
    """Reference field opening at ``pos``, preferring the first-level one."""
    best = None
    for f in fields:
        if int(f['start']) != pos:
            continue
        if best is None or (best.get('parent') is not None
                            and f.get('parent') is None):
            best = f
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(prog='failure_analysis', description=__doc__)
    ap.add_argument('--data', default=None)
    ap.add_argument('--tol', type=int, default=0)
    ap.add_argument('--model-dir', default=None)
    ap.add_argument('--pool-w', type=float, default=None,
                    help='fusion weight w (default: config, 0.7)')
    ap.add_argument('--min-support', type=int, default=30,
                    help='reference boundaries a field needs to be listed')
    ap.add_argument('--json', default='outputs/failure_modes.json')
    a = ap.parse_args(argv)

    cfg = load_config()
    over = {} if a.pool_w is None else {'pool_w': a.pool_w}
    eng = C.make_engine(cfg, model_dir=a.model_dir, **over)
    ref = C.make_ref(cfg)

    by_w = defaultdict(lambda: [0, 0])
    by_bit = defaultdict(lambda: [0, 0])
    by_depth = defaultdict(lambda: [0, 0])
    by_len = defaultdict(lambda: [0, 0])
    by_fam = defaultdict(lambda: [0, 0])
    by_proto = defaultdict(lambda: [0, 0])
    by_field = defaultdict(lambda: [0, 0])
    fp_dist = Counter()
    n_msg = 0

    for pcap in C.capture_paths(a.data):
        proto = Path(pcap).stem
        for rec in C.predict_capture(eng, pcap, cfg, ref):
            n_msg += 1
            gt = set(rec['gt'])
            preds = {int(p) for p in rec['cuts']}
            lb = length_bucket(len(rec['msg']))
            fam = C.FAMILY.get(proto, 'other')
            for b in gt:
                f = field_at(rec['fields'], b)
                hit = 1 if b in preds else 0
                w = (int(f['end']) - int(f['start'])) if f else None
                bit = bool(f and f.get('bit_width') is not None
                           and int(f['bit_width']) < 8)
                depth = 'nested' if (f and f.get('parent')) else 'top'
                for acc, key in ((by_w, width_bucket(w)),
                                 (by_bit, 'bit field' if bit
                                  else 'byte-aligned'),
                                 (by_depth, depth), (by_len, lb),
                                 (by_fam, fam), (by_proto, proto),
                                 (by_field, (f or {}).get('name') or '?')):
                    acc[key][0] += hit
                    acc[key][1] += 1
            for p in preds - gt:
                k = str(min(abs(p - g) for g in gt)) if gt else 'no-gt'
                fp_dist[k if k in ('1', '2', '3', '4') else '>=5'] += 1

    def table(acc, order=None):
        keys = order or sorted(acc, key=lambda k: -acc[k][1])
        return [{'key': k, 'hit': acc[k][0], 'total': acc[k][1],
                 'recall': round(acc[k][0] / acc[k][1], 4)}
                for k in keys if k in acc]

    res = {
        'tol': a.tol, 'n_messages': n_msg,
        'by_field_width': table(by_w, ['1 B', '2 B', '3-4 B', '5-8 B', '>8 B']),
        'by_bit_field': table(by_bit, ['byte-aligned', 'bit field']),
        'by_depth': table(by_depth, ['top', 'nested']),
        'by_msg_len': table(by_len, ['<=16 B', '17-32 B', '33-64 B', '>64 B']),
        'by_family': table(by_fam),
        'by_protocol': table(by_proto),
        'fp_distance': dict(sorted(fp_dist.items())),
        'worst_fields': sorted(
            [{'field': k, 'hit': v[0], 'total': v[1],
              'recall': round(v[0] / v[1], 4)}
             for k, v in by_field.items() if v[1] >= a.min_support],
            key=lambda r: r['recall'])[:20],
    }

    print('messages: %d' % n_msg)
    for key in ('by_field_width', 'by_bit_field', 'by_depth', 'by_msg_len',
                'by_family'):
        print('\n== %s ==' % key)
        for r in res[key]:
            print('  %-24s %6d / %-6d  R=%.4f'
                  % (r['key'], r['hit'], r['total'], r['recall']))
    print('\n== fp distance (bytes) ==', res['fp_distance'])
    print('\n== weakest fields ==')
    for r in res['worst_fields'][:10]:
        print('  %-28s %4d / %-5d  R=%.4f'
              % (r['field'], r['hit'], r['total'], r['recall']))

    C.write_json(a.json, res)


if __name__ == '__main__':
    main()
