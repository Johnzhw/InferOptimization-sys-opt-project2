# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hotness tracking for prefix-tree nodes (HotPrefix, PACMMOD 250 §3.2).

Independent implementation from the paper's description (no author code).
Two interchangeable backends:

- ``CuckooHotnessTracker``: the paper's hotness-aware cuckoo filter
  (§3.2.1, Fig.3): a hash table of n buckets x 4 slots; each entry stores
  (fingerprint, clock, frequency, depth) packed in four 8-bit fields.
  Insert uses f = fingerprint(x), b1 = hash(f), b2 = b1 XOR hash(f)
  (Eqs. 2-4) with cuckoo relocation on full buckets; entries start with
  clock = max_age, frequency = 1 (§3.2.1). Reuse sets frequency += 1 and
  clock = max_age (§3.2.2 Step 1); periodic aging decrements every clock
  by one with a floor of 0 (§3.2.2 Step 2).

- ``ExactHotnessTracker``: bounded exact dictionary with the same
  interface and semantics, used as a test reference to cross-check the
  approximate filter (never used in production paths).

Differences from the paper (documented in mechanism_mapping.md §1.1):
- The hash function is a 64-bit integer mix over the block hash bytes
  instead of MurmurHash (choice of hash does not change semantics).
- 8-bit fields saturate at 255 (frequency) as implied by the paper's
  field widths.
- On insertion failure (filter full), the coldest kicked entry is
  dropped (the paper shuffles unused entries to host memory; we bound
  memory explicitly).
- Aging is driven by the scheduler at request boundaries instead of a
  dedicated host thread (single-threaded equivalence).

Node identity: the vLLM chained block hash, which is exactly the
content identity of the path from the root to the node. Depth is the
block index within the request (block-aligned logical prefix node).
"""

from collections import OrderedDict
from typing import Optional

import numpy as np

# Field layout per entry: [fingerprint, clock, frequency, depth], uint8.
FP, CLOCK, FREQ, DEPTH = 0, 1, 2, 3
EMPTY_FP = 0  # fingerprint value reserved to mark empty slots.

_MASK64 = (1 << 64) - 1


def _mix64(x: int) -> int:
    """splitmix64 finalizer (independent of PYTHONHASHSEED)."""
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _node_hash_int(node_hash: bytes) -> int:
    # Block hashes are >= 16B; fold first and last 8 bytes.
    h = int.from_bytes(node_hash[:8], "little")
    t = int.from_bytes(node_hash[-8:], "little")
    return _mix64(h ^ _mix64(t))


class HotnessRecord(tuple):
    """(frequency, clock, depth) view of one node."""


class ExactHotnessTracker:
    """Bounded exact hotness table (test reference; same semantics).

    Capacity is enforced by evicting the coldest entry
    (min frequency*clock, tie-break oldest insert).
    """

    def __init__(self, capacity: int, max_age: int = 255):
        self.capacity = capacity
        self.max_age = max_age
        # node_hash(int) -> [freq, clock, depth]
        self._table: OrderedDict[int, list] = OrderedDict()
        self.aging_calls = 0
        self.evictions = 0

    def _key(self, node_hash: bytes) -> int:
        return _node_hash_int(node_hash)

    def insert(self, node_hash: bytes, depth: int) -> None:
        k = self._key(node_hash)
        if k in self._table:
            rec = self._table[k]
            rec[0] = min(rec[0] + 1, 255)  # treat re-insert as reuse
            rec[1] = self.max_age
            return
        if len(self._table) >= self.capacity:
            coldest = min(self._table.items(), key=lambda kv: kv[1][0] * kv[1][1])
            del self._table[coldest[0]]
            self.evictions += 1
        self._table[k] = [1, self.max_age, depth & 0xFF]

    def reuse(self, node_hash: bytes) -> bool:
        rec = self._table.get(self._key(node_hash))
        if rec is None:
            return False
        rec[0] = min(rec[0] + 1, 255)
        rec[1] = self.max_age
        return True

    def get(self, node_hash: bytes) -> Optional[tuple]:
        rec = self._table.get(self._key(node_hash))
        return (rec[0], rec[1], rec[2]) if rec is not None else None

    def hotness(self, node_hash: bytes) -> int:
        rec = self.get(node_hash)
        return rec[0] * rec[1] if rec else 0

    def age(self) -> None:
        for rec in self._table.values():
            if rec[1] > 0:
                rec[1] -= 1
        self.aging_calls += 1

    def __len__(self) -> int:
        return len(self._table)


class CuckooHotnessTracker:
    """Hotness-aware cuckoo filter (paper §3.2.1 structure).

    Args:
        num_buckets: number of buckets; each bucket has 4 slots.
        max_age: initial/reset clock value (paper uses 255).
        max_kick: relocation attempts before dropping the coldest entry.
        fp_bits: fingerprint width in bits. The paper's evaluated config
            uses 8-bit fingerprints (§5.2); with our workloads' distinct
            prefix-node counts (hundreds of thousands) an 8-bit
            fingerprint conflates distinct nodes far too often, so the
            default is 16-bit (documented deviation; clock/frequency/
            depth remain 8-bit as in the paper).
    """

    SLOTS = 4

    def __init__(self, num_buckets: int, max_age: int = 255, max_kick: int = 16,
                 fp_bits: int = 16):
        assert num_buckets > 0 and (num_buckets & (num_buckets - 1)) == 0, \
            "num_buckets must be a power of two"
        self.num_buckets = num_buckets
        self.bucket_mask = num_buckets - 1
        self.max_age = max_age & 0xFF
        self.max_kick = max_kick
        self.fp_bits = fp_bits
        self.fp_mask = (1 << fp_bits) - 1
        self.fp_dtype = np.uint16 if fp_bits > 8 else np.uint8
        # fp field stored in a separate wider array; 8-bit fields together.
        self._fp = np.zeros((num_buckets, self.SLOTS), dtype=self.fp_dtype)
        self._meta = np.zeros((num_buckets, self.SLOTS, 3), dtype=np.uint8)
        self._count = 0
        self.kicks = 0
        self.dropped = 0
        self.aging_calls = 0
        self._aging_ops_ns = 0

    # ------------------------------------------------------------------
    # Bucket addressing (Eqs. 2-4)
    # ------------------------------------------------------------------

    def _addresses(self, node_hash: bytes) -> tuple:
        h = _node_hash_int(node_hash)
        fp = (h & self.fp_mask) or 1  # reserve 0 for empty
        h2 = _mix64(fp)
        b1 = h2 & self.bucket_mask
        b2 = (b1 ^ h2) & self.bucket_mask
        return fp, b1, b2

    def _write(self, b: int, s: int, fp: int, clock: int, freq: int, depth: int):
        self._fp[b, s] = fp
        self._meta[b, s, 0] = clock & 0xFF
        self._meta[b, s, 1] = freq & 0xFF
        self._meta[b, s, 2] = depth & 0xFF
        self._count = -1  # invalidate cached count

    # ------------------------------------------------------------------
    # Public API (same as ExactHotnessTracker)
    # ------------------------------------------------------------------

    def _find(self, fp: int, b1: int, b2: int):
        for b in (b1, b2):
            for s in range(self.SLOTS):
                if self._fp[b, s] == fp:
                    return b, s
        return None

    def insert(self, node_hash: bytes, depth: int) -> None:
        fp, b1, b2 = self._addresses(node_hash)
        # Same fingerprint already in a candidate bucket: treat as the same
        # node (fingerprint approximation) and refresh it.
        hit = self._find(fp, b1, b2)
        if hit is not None:
            b, s = hit
            freq = min(int(self._meta[b, s, 1]) + 1, 255)
            depth = int(self._meta[b, s, 2])
            self._write(b, s, fp, self.max_age, freq, depth)
            return
        # Try an empty slot in either candidate bucket.
        for b in (b1, b2):
            for s in range(self.SLOTS):
                if self._fp[b, s] == 0:
                    self._write(b, s, fp, self.max_age, 1, depth)
                    return
        # Both buckets full: cuckoo relocation (Eq. 4 alternate bucket).
        b = b1 if (b1 ^ fp) & 1 else b2
        cur_fp, cur_meta = fp, (self.max_age, 1, depth & 0xFF)
        for _ in range(self.max_kick):
            s = int((_mix64(cur_fp) >> 8) % self.SLOTS)
            old_fp = int(self._fp[b, s])
            old_meta = tuple(int(x) for x in self._meta[b, s])
            self._write(b, s, cur_fp, *cur_meta)
            self.kicks += 1
            if old_fp == 0:
                return
            alt = (b ^ _mix64(old_fp)) & self.bucket_mask
            placed = False
            for s2 in range(self.SLOTS):
                if self._fp[alt, s2] == 0:
                    self._write(alt, s2, old_fp, *old_meta)
                    placed = True
                    break
            if placed:
                return
            b = alt
            cur_fp, cur_meta = old_fp, old_meta
        # Exceeded kicks: drop the displaced entry (bounded memory).
        self.dropped += 1

    def reuse(self, node_hash: bytes) -> bool:
        fp, b1, b2 = self._addresses(node_hash)
        hit = self._find(fp, b1, b2)
        if hit is None:
            return False
        b, s = hit
        freq = min(int(self._meta[b, s, 1]) + 1, 255)
        self._meta[b, s, 1] = freq
        self._meta[b, s, 0] = self.max_age
        return True

    def get(self, node_hash: bytes) -> Optional[tuple]:
        fp, b1, b2 = self._addresses(node_hash)
        hit = self._find(fp, b1, b2)
        if hit is None:
            return None
        b, s = hit
        m = self._meta[b, s]
        return (int(m[1]), int(m[0]), int(m[2]))

    def age(self) -> None:
        import time

        t0 = time.perf_counter_ns()
        clocks = self._meta[:, :, 0]
        np.subtract(clocks, 1, out=clocks, where=clocks > 0)
        self.aging_calls += 1
        self._aging_ops_ns += time.perf_counter_ns() - t0

    def hotness(self, node_hash: bytes) -> int:
        rec = self.get(node_hash)
        return rec[0] * rec[1] if rec else 0

    def __len__(self) -> int:
        if self._count < 0:
            self._count = int(np.count_nonzero(self._fp))
        return self._count
