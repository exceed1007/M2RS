# Data

## Test captures (`data/test_protocol/`)

Twelve captures, used for the evaluation in Section 5.2, come from public collections.
Reference boundaries are produced at run time by tshark, so the captures are
the only data needed to reproduce the per-protocol table.

| Capture | Protocol | Family | Source |
|---|---|---|---|
| `modbus.pcap` | Modbus/TCP | fieldbus SCADA | Wireshark SampleCaptures |
| `dnp3.pcap` | DNP3 | fieldbus SCADA | public ICS collection |
| `IEC104.pcap` | IEC 60870-5-104 | fieldbus SCADA | Wireshark SampleCaptures |
| `S7comm.pcap` | S7comm | fieldbus SCADA | public ICS collection |
| `enip.pcap` | EtherNet/IP and CIP | industrial Ethernet | ICS-Pcaps |
| `omron.pcap` | OMRON FINS | industrial Ethernet | ICS-Pcaps |
| `mms.pcap` | IEC 61850 MMS | substation automation | ICS-Pcaps |
| `bacnet.pcap` | BACNet/IP | building automation | Wireshark SampleCaptures |
| `hart_ip.pcap` | HART-IP | process instrumentation | Wireshark SampleCaptures |
| `DCERPC.pcap` | DCERPC | RPC middleware | Wireshark SampleCaptures |
| `MQTT.pcap` | MQTT | IoT messaging | Wireshark SampleCaptures |
| `snmp.pcap` | SNMP | network management | `traces-set1b` |

SNMP is not an industrial protocol and is kept as the negative end of the
applicability boundary discussed in Section 5.2.6: its variable bindings are
recursively encoded, and a partition of the message cannot represent them.


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
