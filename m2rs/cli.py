"""Command line entry points: ``evaluate`` and ``predict``."""

import argparse
import json
from pathlib import Path

import yaml

from .dataset import TsharkReference, load_sessions
from .engine import M2RSBoundaryEngine
from .metrics import Scoreboard, match_positions

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config' / 'config.yaml'
PCAP_SUFFIXES = ('.pcap', '.pcapng', '.cap')


def load_config(path=None) -> dict:
    with open(path or DEFAULT_CONFIG, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def pcap_files(data_dir):
    d = Path(data_dir)
    if d.is_file():
        return [d]
    out = [p for p in sorted(d.iterdir()) if p.suffix.lower() in PCAP_SUFFIXES]
    if not out:
        raise SystemExit('no capture file found in %s' % d)
    return out


def evaluate(data_dir, config=None, tol: int = 0, model_dir: str = None,
             verbose: bool = True) -> dict:
    """Evaluate the engine on every capture of ``data_dir``.

    Each capture is treated as one protocol, matching the per-protocol table
    in Section 5.2.1.
    """
    cfg = config or load_config()
    eng = M2RSBoundaryEngine(cfg, model_dir=model_dir)
    if not eng.scorer.ready:
        raise SystemExit('no scorer model found under %s' % (model_dir or 'models/mmfi'))
    if eng.scorer.gb is None:
        print('[warn] boundary_scorer.joblib not found: falling back to the '
              'logistic-regression scorer, whose scores are not calibrated '
              'for theta, so the numbers below will not match Section 5.2. '
              'Run "python -m m2rs.train" to rebuild the gradient-boosting '
              'model.')
    ref = TsharkReference(cfg['dataset'].get('tshark', 'tshark'),
                          int(cfg['dataset'].get('tshark_timeout', 300)))
    board = Scoreboard()

    for pcap in pcap_files(data_dir):
        frame_fields, payload_info = ref.load(str(pcap))
        sessions = load_sessions(str(pcap),
                                 float(cfg['dataset'].get('ascii_threshold', 0.4)))
        for sess in sessions:
            preds = eng.predict_messages(sess.messages)
            for i, msg in enumerate(sess.messages):
                fields = ref.fields_for_frames(sess.frames[i], frame_fields,
                                               payload_info, len(msg))
                gt = ref.positions(fields, len(msg))
                tp, fp, fn = match_positions(preds[i], gt, tol)
                board.add(pcap.stem, tp, fp, fn)

    result = board.summary()
    if verbose:
        print('\n'.join(board.format()))
        m = result['micro']
        print('micro-average F1 = %.4f | macro-average F1 = %.4f'
              % (m['F1'], result['macro'].get('F1', 0.0)))
    return result


def predict(pcap: str, config=None, model_dir: str = None) -> dict:
    """Print the inferred template of every length stratum of one capture."""
    cfg = config or load_config()
    eng = M2RSBoundaryEngine(cfg, model_dir=model_dir)
    out = {}
    for sess in load_sessions(str(pcap),
                              float(cfg['dataset'].get('ascii_threshold', 0.4))):
        preds = eng.predict_messages(sess.messages)
        for msg, cuts in zip(sess.messages, preds):
            out.setdefault(len(msg), {}).setdefault(tuple(cuts), 0)
            out[len(msg)][tuple(cuts)] += 1
    for length in sorted(out):
        best = max(out[length].items(), key=lambda kv: kv[1])
        print('L=%-5d messages=%-5d template=%s'
              % (length, sum(out[length].values()), list(best[0])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='m2rs', description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    pe = sub.add_parser('evaluate', help='per-protocol P/R/F1 table')
    pe.add_argument('-d', '--data', default=str(ROOT / 'data' / 'test_protocol'))
    pe.add_argument('-c', '--config', default=None)
    pe.add_argument('--tol', type=int, default=0)
    pe.add_argument('--model-dir', default=None)
    pe.add_argument('--json', default=None, help='write the result as JSON')
    pe.add_argument('--quiet', action='store_true')

    pp = sub.add_parser('predict', help='show decoded templates of one capture')
    pp.add_argument('pcap')
    pp.add_argument('-c', '--config', default=None)
    pp.add_argument('--model-dir', default=None)

    a = ap.parse_args(argv)
    if a.cmd == 'evaluate':
        res = evaluate(a.data, load_config(a.config), a.tol, a.model_dir,
                       verbose=not a.quiet)
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1),
                                    encoding='utf-8')
        return res
    predict(a.pcap, load_config(a.config), a.model_dir)
    return None


if __name__ == '__main__':
    main()
