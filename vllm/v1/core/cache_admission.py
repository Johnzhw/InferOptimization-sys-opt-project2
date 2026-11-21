# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Depth-aware cache admission (Buoy, PACMMOD 263 §4.3).

Independent implementation from the paper's description. Buoy §4.3 states
admission operates independently of the replacement policy, so this module
is shared by the GPU-only combination modes (lru_buoy_admission /
arc_buoy_admission) and the full buoy mode.

Mechanics (§4.3):
- A depth threshold separates shallow prefixes (admitted by default) from
  deep prefixes (rejected by default).
- Rejected prefix chunk IDs enter a bounded FIFO ghost.
- A deep prefix that reappears and hits the ghost is admitted regardless
  of depth (reuse-proven override).
- The threshold is the cumulative mean of *reuse* depths only (O(1)
  running mean), not of all accessed depths, and not an EMA.

Engineering assumptions (documented in mechanism_mapping.md §2.2):
- Depth unit = block index within the request (block-aligned nodes).
- Integer threshold = floor of the running mean.
- Before any reuse is observed the threshold is +inf (admit all) so the
  estimator converges from real reuse data; a sensitivity unit test
  covers this choice.
- Ghost entries are not consumed on override (they age out by FIFO),
  matching Algorithm 1 line 9 which only queries existence.

A reuse event is recorded whenever a cached chunk is hit by a later
request (GPU hit, or CPU hit that leads to a successful reload).
"""

from collections import OrderedDict
from typing import Optional


class BuoyAdmission:
    """Reusable depth-aware admission controller (Buoy §4.3)."""

    def __init__(self, ghost_capacity: int, enabled: bool = True):
        self.enabled = enabled
        self._ghost: OrderedDict[bytes, None] = OrderedDict()
        self._ghost_capacity = ghost_capacity
        # running mean of reuse depths
        self._reuse_count = 0
        self._reuse_depth_sum = 0
        # stats
        self.admits = 0
        self.rejects = 0
        self.ghost_overrides = 0
        self.ghost_hits_shallow = 0

    # ------------------------------------------------------------------
    @property
    def threshold(self) -> Optional[int]:
        """Current depth threshold; None = admit everything (no data yet)."""
        if self._reuse_count == 0:
            return None
        return self._reuse_depth_sum // self._reuse_count

    def admit(self, node_hash: bytes, depth: int) -> bool:
        """Decide whether a newly computed chunk may be retained across
        requests. Rejection here never drops KV needed by a running
        request; it only prevents cross-request retention."""
        if not self.enabled:
            self.admits += 1
            return True
        if node_hash in self._ghost:
            # Reuse-proven override: admit regardless of depth.
            self.ghost_overrides += 1
            self.admits += 1
            return True
        th = self.threshold
        if th is None or depth <= th:
            self.admits += 1
            return True
        # Deep prefix rejected by default; record in the bounded ghost.
        self.rejects += 1
        self._ghost[node_hash] = None
        while len(self._ghost) > self._ghost_capacity:
            self._ghost.popitem(last=False)
        return False

    def on_reuse(self, depth: int) -> None:
        """Record one reuse event (a chunk hit at the given depth)."""
        self._reuse_count += 1
        self._reuse_depth_sum += depth

    def reset(self) -> None:
        self._ghost.clear()
        self._reuse_count = 0
        self._reuse_depth_sum = 0
        self.admits = self.rejects = 0
        self.ghost_overrides = 0
        self.ghost_hits_shallow = 0

    def snapshot(self) -> dict:
        return {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "reuse_events": self._reuse_count,
            "admits": self.admits,
            "rejects": self.rejects,
            "ghost_overrides": self.ghost_overrides,
            "ghost_size": len(self._ghost),
            "ghost_capacity": self._ghost_capacity,
        }
