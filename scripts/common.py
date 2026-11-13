"""Helpers shared by the Section 5 experiment scripts.

Every script under ``scripts/`` needs the same three things: the test captures
as (message, reference fields) pairs, a P/R/F1 accumulation that is identical
to ``m2rs.cli.evaluate``, and a way to refit the discriminator on a subset of
the training corpus.  Keeping them here avoids three slightly different
implementations of the same metric.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from m2rs.cli import load_config, pcap_files          # noqa: E402
from m2rs.dataset import TsharkReference, load_sessions  # noqa: E402
from m2rs.engine import M2RSBoundaryEngine            # noqa: E402
from m2rs.metrics import match_positions, prf         # noqa: E402

# Protocol -> family, used by the failure decomposition (Section 5.2.6).
FAMILY = {
    'modbus': 'fieldbus SCADA', 'dnp3': 'fieldbus SCADA',
    'IEC104': 'fieldbus SCADA', 'S7comm': 'fieldbus SCADA',
    'enip': 'industrial Ethernet', 'omron': 'industrial Ethernet',
    'mms': 'substation automation', 'bacnet': 'building automation',
    'hart_ip': 'process instrumentation', 'DCERPC': 'RPC middleware',
    'MQTT': 'IoT messaging', 'snmp': 'network management',
}


def capture_paths(data_dir=None) -> List[Path]:
    return pcap_files(data_dir or (ROOT / 'data' / 'test_protocol'))


def engine_config(cfg: dict, **over) -> dict:
    """Config dict with the boundary parameters overridden."""
    m = dict(cfg['boundary']['m2rs'])
    m.update(over)
    return {'dataset': cfg.get('dataset', {}), 'boundary': {'m2rs': m}}


def make_engine(cfg: dict, model_dir=None, **over) -> M2RSBoundaryEngine:
    return M2RSBoundaryEngine(engine_config(cfg, **over), model_dir=model_dir)


def make_ref(cfg: dict) -> TsharkReference:
    d = cfg.get('dataset') or {}
    return TsharkReference(d.get('tshark', 'tshark'),
                          int(d.get('tshark_timeout', 300)))


def predict_capture(engine, pcap, cfg: dict,
                    ref: TsharkReference) -> List[dict]:
    """Predict one capture; return one record per message.

    Each record holds the message bytes, the inferred cuts, the reference
    fields (name, start, end, bit width, parent) and the reference boundary
    positions, so that the failure and variable-length analyses can attribute
    every hit and every false positive to a field.
    """
    thr = float((cfg.get('dataset') or {}).get('ascii_threshold', 0.4))
    frame_fields, payload_info = ref.load(str(pcap))
    out: List[dict] = []
    for sess in load_sessions(str(pcap), thr):
        preds = engine.predict_messages(sess.messages)
        for i, msg in enumerate(sess.messages):
            fields = ref.fields_for_frames(sess.frames[i], frame_fields,
                                           payload_info, len(msg))
            out.append({
                'proto': Path(pcap).stem,
                'msg': msg,
                'cuts': list(preds[i]),
                'fields': fields,
                'gt': ref.positions(fields, len(msg)),
            })
    return out


def score_records(engine, records: Sequence[dict], tol: int = 0) -> List[int]:
    """(tp, fp, fn) of a list of predict_capture records."""
    acc = [0, 0, 0]
    for r in records:
        t = match_positions(r['cuts'], r['gt'], tol)
        acc = [acc[i] + t[i] for i in range(3)]
    return acc


def evaluate(engine, cfg: dict, data_dir=None, tol: int = 0,
             ref: Optional[TsharkReference] = None,
             keep_records: bool = False) -> dict:
    """Per-protocol P/R/F1, micro and macro average.

    Mirrors ``m2rs.cli.evaluate`` so that a number quoted in a table can be
    produced by either entry point.
    """
    ref = ref or make_ref(cfg)
    per: Dict[str, List[int]] = {}
    total = [0, 0, 0]
    records = []
    for pcap in capture_paths(data_dir):
        recs = predict_capture(engine, pcap, cfg, ref)
        acc = score_records(engine, recs, tol)
        per[Path(pcap).stem] = acc
        total = [total[i] + acc[i] for i in range(3)]
        if keep_records:
            records.extend(recs)
    out = {
        'per_protocol': {k: prf(*v) for k, v in sorted(per.items())},
        'micro': prf(*total),
        'macro': {},
    }
    if out['per_protocol']:
        for k in ('P', 'R', 'F1'):
            out['macro'][k] = sum(v[k] for v in out['per_protocol'].values()) \
                / len(out['per_protocol'])
    if keep_records:
        out['records'] = records
    return out


def prior_from_labels(groups: Sequence[str], y: Sequence[int],
                      max_w: int = 64, eps: float = 1e-5) -> dict:
    """Field-width prior implied by the labels of a (sub)corpus.

    Same quantity as ``m2rs.train.width_prior``, recomputed from labels only,
    which is what a subset of the corpus can provide without rebuilding it.
    """
    order = defaultdict(list)
    for g, lab in zip(groups, y):
        order[g].append(int(lab))
    mid, last = Counter(), Counter()
    for labs in order.values():
        length = len(labs) + 1
        edges = [0] + [j + 1 for j, v in enumerate(labs) if v == 1] + [length]
        for k in range(len(edges) - 1):
            w = edges[k + 1] - edges[k]
            (last if k == len(edges) - 2 else mid)[w] += 1

    def norm(cnt):
        tot = sum(cnt.values()) or 1
        p = {str(w): max(cnt.get(w, 0) / tot, eps) for w in range(1, max_w + 1)}
        s = sum(p.values())
        return {k: v / s for k, v in p.items()}

    return {'max_len': max_w, 'p_mid': norm(mid), 'p_last': norm(last),
            'tail_rate_mid': 0.35, 'tail_rate_last': 0.02}


def refit(X, y, names, groups, out_dir, seed: int = 7,
          prior: Optional[dict] = None, max_w: int = 64) -> Path:
    """Refit the discriminator (and the prior) into ``out_dir``."""
    from m2rs.train import fit

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fit(X, y, list(names), list(groups), out_dir, seed=seed, with_gb=True)
    p = prior if prior is not None else prior_from_labels(groups, y, max_w)
    (out_dir / 'len_prior.json').write_text(json.dumps(p, indent=1),
                                           encoding='utf-8')
    return out_dir


def pick_groups(groups: Sequence[str], ratio: float, seed: int) -> List[str]:
    """A random subset of whole strata (strata are the annotation unit)."""
    uniq = sorted(set(groups))
    rng = np.random.RandomState(seed)
    k = max(1, int(round(len(uniq) * float(ratio))))
    return list(rng.permutation(uniq)[:k])


def csv_numbers(text: str) -> List[float]:
    return [float(x) for x in str(text).split(',') if x.strip()]


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1), encoding='utf-8')
    print('written to %s' % path)
