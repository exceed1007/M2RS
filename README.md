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

Micro-averaged over the twelve test protocols, with the settings of Section 5.1
(`theta = -2.8`, `w = 0.7`). Reference boundaries are the first-level fields
reported by tshark, matched at zero tolerance.

| Protocol | Family | #msgs | #gt | P | R | F1 |
|---|---|---|---|---|---|---|
| Modbus/TCP | fieldbus SCADA | 64 | 320 | 0.424 | 1.000 | 0.595 |
| DNP3 | fieldbus SCADA | 117 | 913 | 0.462 | 0.846 | 0.598 |
| IEC 60870-5-104 | fieldbus SCADA | 63 | 516 | 0.773 | 0.758 | 0.765 |
| S7comm | fieldbus SCADA | 302 | 2382 | 0.512 | 0.804 | 0.625 |
| EtherNet/IP | industrial Ethernet | 592 | 7092 | 0.459 | 0.958 | 0.621 |
| OMRON FINS | industrial Ethernet | 245 | 3323 | 0.940 | 0.740 | 0.828 |
| IEC 61850 MMS | substation automation | 259 | 2375 | 0.902 | 0.606 | 0.725 |
| BACNet | building automation | 100 | 1696 | 0.596 | 0.751 | 0.665 |
| HART-IP | process instrumentation | 49 | 598 | 0.692 | 0.470 | 0.560 |
| DCERPC | RPC middleware | 318 | 3657 | 0.532 | 0.609 | 0.568 |
| MQTT | IoT messaging | 97 | 344 | 0.858 | 0.846 | 0.852 |
| SNMP | network management | 631 | 5254 | 0.318 | 0.397 | 0.353 |
| **Micro average (12)** | | **2837** | **28470** | **0.515** | **0.711** | **0.598** |

Micro-averaged F1 over the eight protocols for which published baseline results
exist is 0.618 (P = 0.544, R = 0.716); macro-averaged F1 over the twelve is
0.646. SNMP is included deliberately: it is a management protocol with
recursively encoded data, and it marks where the method stops working
(Section 5.2.6) rather than flattering the average.

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

Reproduce the experiments of Section 5.2:

```bash
python scripts/leave_one_out.py --folds 7                # Table 2: leave-one-source-out
python scripts/ablation.py --features                    # Tables 3-4: mechanisms, feature families
python scripts/variable_length.py --pool-w 1.0           # Table 5: variable-length fields
python scripts/annotation_budget.py                      # Table 6: annotation budget
python scripts/failure_analysis.py --pool-w 1.0          # Table 7: failure decomposition
```

Tables 5 and 7 in the manuscript were computed with the fusion weight at 1.0;
`--pool-w 1.0` reproduces them, while the other experiments use the 0.7 of
Section 5.1.

Table 1 comes straight from `evaluate`; the other five need the
known-protocol corpus (`data/known_protocol/`), which is not shipped — see
[Data](#data). They build the corpus once and then refit per fold or per
budget, so a full run is dominated by tshark, not by training.

`--json` on any of them writes the numbers in machine-readable form, and
`--data` points the evaluation at another capture directory.

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

`data/test_protocol/` holds the twelve captures used for evaluation, taken from
public sources (Wireshark SampleCaptures and the ICS capture collections); the
family of each protocol is the one used in Section 5.2.6. The known-protocol
corpus used for training is large (~1 GB) and is not distributed here;
`data/README.md` lists where to obtain it and how it is laid out.

## Repository layout

```
m2rs/features.py    42 structural features per candidate cut point (Section 4.3)
m2rs/engine.py      scorer, width prior, fusions and MAP decoding (Sections 4.2-4.5)
m2rs/dataset.py     capture -> messages + frames, tshark PDML -> reference boundaries
m2rs/metrics.py     one-to-one boundary matching, P/R/F1, micro/macro averaging
m2rs/train.py       corpus construction and model fitting
m2rs/cli.py         evaluate / predict entry points

scripts/common.py            shared evaluation and refitting helpers
scripts/ablation.py          mechanism and feature-family ablations (Tables 3-4)
scripts/leave_one_out.py     leave-one-source-out validation (Table 2)
scripts/variable_length.py   variable-length field subtask (Table 5)
scripts/annotation_budget.py accuracy vs. annotated corpus size (Table 6)
scripts/failure_analysis.py  recall by field width, length, family (Table 7)

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
