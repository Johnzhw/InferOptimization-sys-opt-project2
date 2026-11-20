# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selective CPU admission for evicted KV (HotPrefix §3.3.2).

Independent implementation from the paper's description. Two-stage filter
for GPU->CPU offload candidates:

1. Frequency threshold: nodes whose tracked access frequency is below the
   threshold (paper value 10, §5.2) are discarded immediately — the
   D2H transfer never starts (the saving is real, not index-only).
2. Hotness comparison (Eq. 6: hotness = frequency * clock): when the CPU
   tier has no free slot, the candidate is admitted only if its hotness
   exceeds the hotness of the least critical CPU resident candidate; the
   comparison uses the *actual* victim candidate's tracked hotness.

Engineering assumption (paper does not state the free-slot case): when
the CPU tier has free capacity, only the frequency threshold applies.

The victim-side hook is intentionally injectable so different CPU-side
replacement structures can supply their real coldest candidate (the
hotprefix mode uses a hotness-ordered CPU tier; the arc_hotprefix_admission
mode keeps an LRU CPU tier but still compares against the LRU victim's
tracked hotness — difference documented in mechanism_mapping.md §3).
"""

from typing import Callable, Optional

from vllm.v1.core.hotness_tracker import CuckooHotnessTracker


class HotPrefixAdmission:
    """HotPrefix §3.3.2 selective admission for CPU offload candidates."""

    def __init__(
        self,
        tracker: CuckooHotnessTracker,
        freq_threshold: int = 10,
        enabled: bool = True,
    ):
        self.tracker = tracker
        self.freq_threshold = freq_threshold
        self.enabled = enabled
        # stats
        self.passed_freq = 0
        self.rejected_freq = 0
        self.rejected_hotness = 0
        self.admitted = 0
        self.comparisons = 0

    def hotness(self, node_hash: bytes) -> int:
        return self.tracker.hotness(node_hash)

    def frequency(self, node_hash: bytes) -> int:
        rec = self.tracker.get(node_hash)
        return rec[0] if rec else 0

    def admit(
        self,
        node_hash: bytes,
        cpu_has_free_slot: bool,
        coldest_cpu_hotness: Optional[Callable[[], int]] = None,
    ) -> bool:
        """Decide whether an evicted GPU node's KV may be offloaded.

        Args:
            node_hash: identity of the candidate node.
            cpu_has_free_slot: whether the CPU tier currently has a free
                slot (no victim comparison needed).
            coldest_cpu_hotness: lazy callable returning the tracked
                hotness of the real coldest CPU resident candidate
                (only consulted when the CPU tier is full).
        """
        if not self.enabled:
            self.admitted += 1
            return True
        freq = self.frequency(node_hash)
        if freq < self.freq_threshold:
            self.rejected_freq += 1
            return False
        self.passed_freq += 1
        if not cpu_has_free_slot and coldest_cpu_hotness is not None:
            self.comparisons += 1
            if self.hotness(node_hash) <= coldest_cpu_hotness():
                self.rejected_hotness += 1
                return False
        self.admitted += 1
        return True

    def reset(self) -> None:
        self.passed_freq = 0
        self.rejected_freq = 0
        self.rejected_hotness = 0
        self.admitted = 0
        self.comparisons = 0

    def snapshot(self) -> dict:
        return {
            "freq_threshold": self.freq_threshold,
            "enabled": self.enabled,
            "passed_freq": self.passed_freq,
            "rejected_freq": self.rejected_freq,
            "rejected_hotness": self.rejected_hotness,
            "admitted": self.admitted,
            "comparisons": self.comparisons,
        }
