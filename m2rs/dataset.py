"""Traffic ingestion and reference boundary construction (Section 5.1).

Two responsibilities:

* turn a capture into application-layer messages (TCP sessions are
  reassembled at PSH boundaries, UDP datagrams map one-to-one), together with
  the frame numbers each message came from;
* turn the tshark PDML output into reference boundary positions, i.e. the
  start offset of every first-level application-layer field.

Reference boundaries are *not* the model input: they are only used for
training labels and for evaluation.
"""

import re
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from scapy.all import IP, TCP, UDP, Raw, rdpcap

# Layers that carry no application payload semantics.
BASE_LAYERS = {
    'geninfo', 'frame', 'eth', 'ethertype', 'ip', 'ipv6', 'arp', 'icmp',
    'icmpv6', 'vlan', 'radiotap', 'wlan', 'sll', 'nsh', 'mpls', 'tcp',
    'udp', 'tcp.segments', 'data', 'data.data', 'text',
}

# tshark helper fields kept out of the reference set.
EXCLUDED_PREFIXES = ('frame.', 'tcp.', 'udp.', 'ip.', 'ipv6.', 'eth.',
                     '_ws.', 'expert', 'data.', 'text.')

# A bit-level field is shown as a bitmap, e.g. "..1. .... = Flag: 1".
BIT_PATTERN = re.compile(r'^([01xX.\s]+?)\s*=')

# Ports of the binary protocols used here. They bypass the printable-ASCII
# test, which otherwise rejects MQTT-like protocols that carry text payloads.
BINARY_PORTS = frozenset({
    1883, 8883, 8884,      # MQTT
    102,                   # S7comm
    502,                   # Modbus/TCP
    44818, 2222,           # EtherNet/IP, CIP
    20000,                 # DNP3
    2404,                  # IEC 60870-5-104
    47808,                 # BACnet/IP
    5094,                  # HART-IP
    135,                   # DCERPC
    4840,                  # OPC UA
    34964,                 # PROFINET
})

MIN_MSG_LEN = 4


@dataclass
class Session:
    """Application-layer messages of one transport session."""
    messages: List[bytes] = field(default_factory=list)
    frames: List[List[int]] = field(default_factory=list)


def read_packets(pcap: str) -> List[dict]:
    """Read a capture into plain dicts (frame numbers are 1-based)."""
    out = []
    for idx, pkt in enumerate(rdpcap(pcap)):
        rec = {
            'timestamp': float(pkt.time),
            'frame_number': idx + 1,
            'src_ip': None, 'dst_ip': None,
            'src_port': None, 'dst_port': None,
            'payload': b'', 'seq': 0, 'flags': 0, 'is_tcp': False,
        }
        if IP in pkt:
            rec['src_ip'] = pkt[IP].src
            rec['dst_ip'] = pkt[IP].dst
            if TCP in pkt:
                rec['src_port'] = int(pkt[TCP].sport)
                rec['dst_port'] = int(pkt[TCP].dport)
                rec['seq'] = int(pkt[TCP].seq)
                rec['flags'] = int(pkt[TCP].flags)
                rec['is_tcp'] = True
            elif UDP in pkt:
                rec['src_port'] = int(pkt[UDP].sport)
                rec['dst_port'] = int(pkt[UDP].dport)
        if Raw in pkt:
            rec['payload'] = bytes(pkt[Raw])
        out.append(rec)
    return out


def group_sessions(packets: Sequence[dict]) -> List[List[dict]]:
    """Group packets by bidirectional 5-tuple, ordered by time."""
    buckets: Dict[tuple, List[dict]] = defaultdict(list)
    for p in packets:
        if p['src_ip'] is None or p['dst_ip'] is None:
            continue
        a = (p['src_ip'], p['src_port'])
        b = (p['dst_ip'], p['dst_port'])
        key = (a, b) if a <= b else (b, a)
        buckets[key].append(p)
    return [sorted(v, key=lambda x: x['timestamp']) for v in buckets.values()]


def _reassemble_with_frames(packets: Sequence[dict]
                            ) -> List[Tuple[float, bytes, List[int]]]:
    """Rebuild TCP messages at PSH boundaries, keeping source frame numbers.

    Segments are merged only while sequence numbers are contiguous; a segment
    with the PSH flag terminates a message even if the next one continues the
    sequence, which keeps interleaved request/response traffic split.
    """
    buffers: Dict[tuple, List[dict]] = defaultdict(list)
    done: List[Tuple[float, bytes, List[int]]] = []

    def key_of(p):
        return (p['src_ip'], p['src_port'], p['dst_ip'], p['dst_port'])

    for p in sorted(packets, key=lambda x: x['timestamp']):
        if not p['is_tcp'] or not p['payload']:
            continue
        k = key_of(p)
        if any(e['seq'] == p['seq'] for e in buffers[k]):
            continue                      # retransmission
        buffers[k].append(p)

        pkts = buffers[k]
        if len(pkts) < 2:
            continue
        pkts.sort(key=lambda x: x['seq'])
        consumed = set()
        i, n = 0, len(pkts)
        while i < n:
            end = i
            while (end + 1 < n and
                   pkts[end + 1]['seq'] == pkts[end]['seq'] + len(pkts[end]['payload'])):
                end += 1
            start, j = i, i
            while j <= end:
                if pkts[j]['flags'] & 0x08:
                    seg = pkts[start:j + 1]
                    done.append((seg[0]['timestamp'],
                                 b''.join(x['payload'] for x in seg),
                                 [x['frame_number'] for x in seg]))
                    consumed.update(range(start, j + 1))
                    start = j + 1
                j += 1
            i = end + 1
        if consumed:
            buffers[k] = [p for idx, p in enumerate(pkts) if idx not in consumed]

    for k in list(buffers):
        pkts = sorted(buffers[k], key=lambda x: x['seq'])
        if not pkts:
            continue
        groups, cur = [], [pkts[0]]
        for p in pkts[1:]:
            if p['seq'] == cur[-1]['seq'] + len(cur[-1]['payload']):
                cur.append(p)
            else:
                groups.append(cur)
                cur = [p]
        groups.append(cur)
        for g in groups:
            start = 0
            for j, p in enumerate(g):
                if p['flags'] & 0x08:
                    done.append((g[start]['timestamp'],
                                 b''.join(x['payload'] for x in g[start:j + 1]),
                                 [x['frame_number'] for x in g[start:j + 1]]))
                    start = j + 1
            if start < len(g):
                done.append((g[start]['timestamp'],
                             b''.join(x['payload'] for x in g[start:]),
                             [x['frame_number'] for x in g[start:]]))
        buffers[k] = []

    done.sort(key=lambda x: x[0])
    return done


def _is_binary(packets: Sequence[dict], messages: Sequence[bytes],
               threshold: float) -> bool:
    ports = set()
    for p in packets:
        for k in ('src_port', 'dst_port'):
            if p[k]:
                ports.add(p[k])
    if ports & BINARY_PORTS:
        return True
    total = printable = 0
    for m in list(messages)[:10]:
        total += len(m)
        printable += sum(1 for b in m if 32 <= b <= 126 or b in (9, 10, 13))
    return total == 0 or printable / total <= threshold


def load_sessions(pcap: str, ascii_threshold: float = 0.4) -> List[Session]:
    """Messages and source frames of every binary application session."""
    packets = read_packets(pcap)
    out: List[Session] = []
    for pkts in group_sessions(packets):
        if not pkts:
            continue
        if any(p['is_tcp'] for p in pkts):
            framed = _reassemble_with_frames(pkts)
            all_msgs = [m for _, m, _ in framed]
        else:
            all_msgs = [p['payload'] for p in pkts if p['payload']]
            framed = [(p['timestamp'], p['payload'], [p['frame_number']])
                      for p in pkts if p['payload']]

        msgs = [m for m in all_msgs if m and len(m) >= MIN_MSG_LEN]
        if not msgs or not _is_binary(pkts, msgs, ascii_threshold):
            continue

        # Map every kept message back to the frames it was rebuilt from.
        pool = [(i, f) for i, (_, m, f) in enumerate(framed)
                if m and len(m) >= MIN_MSG_LEN]
        mapping: List[List[int]] = []
        used = set()
        for m in msgs:
            found = None
            for pos, (idx, frames) in enumerate(pool):
                if pos not in used and framed[idx][1] == m:
                    found = (pos, frames)
                    break
            if found is None:
                mapping.append([])
            else:
                used.add(found[0])
                mapping.append(found[1])
        out.append(Session(messages=msgs, frames=mapping))
    return out


class TsharkReference:
    """Reference boundaries from tshark PDML."""

    def __init__(self, tshark: str = 'tshark', timeout: int = 300):
        self.tshark = tshark
        self.timeout = timeout
        self._cache: Dict[str, Tuple[dict, dict]] = {}

    def load(self, pcap: str) -> Tuple[Dict[int, List[dict]], Dict[int, dict]]:
        """Return (frame fields, payload offsets) for one capture."""
        if pcap in self._cache:
            return self._cache[pcap]
        cmd = [self.tshark, '-r', str(pcap), '-T', 'pdml']
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=self.timeout,
                                 encoding='utf-8', errors='replace')
            root = ET.fromstring(res.stdout or '<pdml/>')
        except Exception:
            root = ET.fromstring('<pdml/>')

        frame_fields: Dict[int, List[dict]] = {}
        payload_info: Dict[int, dict] = {}
        for packet in root.findall('packet'):
            fnum = None
            for proto in packet.findall('proto'):
                if proto.get('name') == 'frame':
                    node = proto.find("./field[@name='frame.number']")
                    if node is not None:
                        try:
                            fnum = int(node.get('show'))
                        except (TypeError, ValueError):
                            fnum = None
                    break
            if fnum is None:
                continue
            payload_info[fnum] = self._payload_info(packet)
            frame_fields[fnum] = self._field_intervals(packet)
        self._cache[pcap] = (frame_fields, payload_info)
        return frame_fields, payload_info

    @staticmethod
    def _payload_info(packet: ET.Element) -> dict:
        pos, size = None, 0
        for proto in packet.iter('proto'):
            if proto.get('name') in ('tcp', 'udp', 'data'):
                for f in proto.iter('field'):
                    if f.get('name') in ('tcp.payload', 'udp.payload',
                                         'data.data', 'data'):
                        try:
                            pos = int(f.get('pos'))
                            size = int(f.get('size') or 0)
                        except (TypeError, ValueError):
                            pass
                        break
                if pos is not None and size > 0:
                    break
        if pos is None or size <= 0:
            for proto in packet.iter('proto'):
                if (proto.get('name') or '') in BASE_LAYERS:
                    continue
                try:
                    p = int(proto.get('pos'))
                    s = int(proto.get('size') or 0)
                except (TypeError, ValueError):
                    continue
                if s > 0:
                    pos, size = p, s
                    break
        return {'payload_pos': pos or 0, 'payload_len': size}

    @staticmethod
    def _bit_width(elem: ET.Element) -> Optional[int]:
        m = BIT_PATTERN.match(elem.get('showname') or '')
        if not m:
            return None
        return sum(1 for c in m.group(1) if c in '01xX')

    def _field_intervals(self, packet: ET.Element) -> List[dict]:
        """Byte intervals of application-layer fields, with nesting info."""
        out: List[dict] = []

        def walk(elem: ET.Element, base: bool, parent: Optional[dict]):
            for child in elem:
                if child.tag == 'proto':
                    walk(child, base or (child.get('name') or '') in BASE_LAYERS,
                         parent)
                elif child.tag == 'field':
                    if base:
                        continue
                    name = child.get('name') or ''
                    if any(name.startswith(p) for p in EXCLUDED_PREFIXES):
                        continue
                    if child.get('generated') == 'yes':
                        continue
                    pos, size = child.get('pos'), child.get('size')
                    if pos is None or size is None:
                        continue
                    try:
                        start, n = int(pos), int(size)
                    except (TypeError, ValueError):
                        continue
                    if n <= 0:
                        continue
                    bw = self._bit_width(child)
                    out.append({'name': name, 'start': start,
                                'end': start + n - 1, 'bit_width': bw,
                                'parent': parent})
                    walk(child, base, {
                        'name': name, 'start': start, 'end': start + n - 1,
                        'bit_width': bw} if name else parent)

        walk(packet, False, None)
        return out

    @staticmethod
    def fields_for_frames(frames: Sequence[int],
                          frame_fields: Dict[int, List[dict]],
                          payload_info: Dict[int, dict],
                          msg_len: int) -> List[dict]:
        """Translate frame-local intervals into message-local intervals."""
        fields: List[dict] = []
        offset = 0
        for fnum in frames or []:
            if fnum is None:
                continue
            info = payload_info.get(fnum) or {}
            base = int(info.get('payload_pos') or 0)
            flen = int(info.get('payload_len') or 0)
            for f in frame_fields.get(fnum) or []:
                s = max(0, int(f['start']) - base)
                e = int(f['end']) - base + 1
                if flen > 0:
                    e = min(e, flen)
                if e <= s:
                    continue
                fields.append({'name': f.get('name') or '',
                               'start': offset + s, 'end': offset + e,
                               'bit_width': f.get('bit_width'),
                               'parent': f.get('parent')})
            offset += flen
        return [f for f in fields if f['start'] < msg_len]

    @staticmethod
    def positions(fields: Sequence[dict], msg_len: int,
                  first_level: bool = True) -> List[int]:
        """Boundary positions of first-level fields (offset 0 excluded)."""
        keep = [f for f in fields
                if not first_level or f.get('parent') is None]
        return sorted({int(f['start']) for f in keep
                       if 0 < int(f['start']) < msg_len})
