# Data

## Test captures (`data/test_protocol/`)

Eight captures, used for the evaluation in Section 5.2, come from public collections.
Reference boundaries are produced at run time by tshark, so the captures are
the only data needed to reproduce the per-protocol tables.

## Known-protocol corpus (`data/known_protocol/`, not included)

Training the gradient-boosting discriminator needs a corpus of captures of
*known* protocols. It is about 1 GB and is therefore not versioned here.
Place the captures under `data/known_protocol/<source>/` — any directory
layout works, but keeping one sub-directory per source keeps the sampling in
`m2rs/train.py` diverse. Public sources that can be used:

* Wireshark SampleCaptures — <https://wiki.wireshark.org/SampleCaptures>
* ICS-Pcaps, ICS-Security-Tools, 4SICS ICS Village CTF captures, and the
  `traces-set1b` traces, all distributed as public ICS traffic collections.

Captures that are byte-identical to a file in `data/test_protocol/` are
excluded from training automatically, so the evaluation protocols never leak
into the model.
