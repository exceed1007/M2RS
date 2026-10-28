# M2RS

Reference implementation of **M2RS** (*Multi-Mechanism Robust Segmentation*),
a method for inferring field boundaries in binary network protocols from raw
traffic. Given the application-layer messages of a session, it outputs the byte
offsets at which fields start — no protocol specification and no source code
required.

The pipeline follows the four mechanisms of the paper:

| Mechanism | Section | Module |
|---|---|---|
| Equal-length message stratification and template sharing | 4.2 | `m2rs/engine.py` (`predict_messages`) |
| General-feature-driven boundary discrimination | 4.3 | `m2rs/features.py` |
| Field-width prior and multi-scale evidence fusion | 4.4 | `m2rs/engine.py` (`WidthPrior`, `features_global`) |
| MAP template decoding and consistency check | 4.5 | `m2rs/engine.py` (`decode_map`) |

## Results

Micro-averaged over the eight test protocols, with the settings of Section 5.1
(`theta = -2.8`, `w = 0.7`):

| Protocol | P | R | F1 |
|---|---|---|---|
| BACNet | 0.596 | 0.751 | 0.665 |
| DCERPC | 0.532 | 0.609 | 0.568 |
| DNP3 | 0.462 | 0.846 | 0.598 |
| HART-IP | 0.692 | 0.470 | 0.560 |
| IEC 60870-5-104 | 0.773 | 0.758 | 0.765 |
| Modbus/TCP | 0.424 | 1.000 | 0.595 |
| MQTT | 0.858 | 0.846 | 0.852 |
| S7comm | 0.512 | 0.804 | 0.625 |
| **Micro average** | **0.544** | **0.716** | **0.618** |

MACRO-averaged F1 is 0.653. Reference boundaries are the first-level fields
reported by tshark, matched at zero tolerance.

## Requirements

* Python 3.8+
* [tshark](https://www.wireshark.org/) on `PATH` (only needed for reference
  boundaries, i.e. for training and evaluation — not for inference)
* `pip install -r requirements.txt`

## Usage

The commands below assume `models/mmfi/boundary_scorer.joblib` is present (see
[Model](#model)). Without it the engine falls back to the logistic-regression
scorer and every number below changes.

Evaluate on the bundled test captures:

```bash
python -m m2rs.cli evaluate            # per-protocol table + micro/macro average
python -m m2rs.cli evaluate --tol 1    # tolerate +-1 byte when matching
python -m m2rs.cli evaluate --json out.json
```

Infer the template of one capture:

```bash
python -m m2rs.cli predict data/test_protocol/MQTT.pcap
# L=51    messages=48    template=[1, 2, 4, 16, 17, 18]
```

Use it as a library:

```python
from m2rs import M2RSBoundaryEngine
from m2rs.dataset import load_sessions

engine = M2RSBoundaryEngine()
for session in load_sessions('capture.pcapng'):
    for msg, cuts in zip(session.messages, engine.predict_messages(session.messages)):
        print(len(msg), cuts)
```

Reproduce the ablation study of Section 5.2.3:

```bash
python scripts/ablation.py
python scripts/ablation.py --features   # feature families as well
```

## Model

`models/mmfi/boundary_scorer_lr.json` and `len_prior.json` are shipped, so the
code runs out of the box. The gradient-boosting discriminator that produced the
numbers above (`boundary_scorer.joblib`, ~1.9 MB) is **not** in the repository;
build it from the known-protocol corpus:

```bash
python -m m2rs.train --known-dir /path/to/known_protocol --max-files 200
```

Training takes the captures of known protocols, keeps the strata whose messages
share one reference template, and fits the discriminator plus the field-width
prior. Without the gradient-boosting model the engine falls back to the
logistic-regression parameters, whose scores are not calibrated for `theta`;
`evaluate` warns about this and the numbers will not match the table above.

## Data

`data/test_protocol/` holds the eight captures used for evaluation, taken from
public sources (Wireshark SampleCaptures and the ICS capture collections). The
known-protocol corpus used for training is large (~1 GB) and is not
distributed here; `data/README.md` lists where to obtain it.

## Repository layout

```
m2rs/features.py    42 structural features per candidate cut point (Section 4.3)
m2rs/engine.py      scorer, width prior, fusions and MAP decoding (Sections 4.2-4.5)
m2rs/dataset.py     capture -> messages + frames, tshark PDML -> reference boundaries
m2rs/metrics.py     one-to-one boundary matching, P/R/F1, micro/macro averaging
m2rs/train.py       corpus construction and model fitting
m2rs/cli.py         evaluate / predict entry points
scripts/ablation.py mechanism and feature-family ablations
tests/              unit tests (no capture or tshark needed)
```

## Citation

```bibtex
@article{m2rs2026,
  title  = {Field boundary inference for binary protocols via multi-dimensional
            feature aggregation and multi-mechanism segmentation},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

MIT — see [LICENSE](LICENSE).
