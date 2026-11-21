# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replacement policies for the GPU-resident prefix cache (V1 engine).

This module implements an ARC (Adaptive Replacement Cache; Megiddo & Modha,
FAST '03) replacement policy for the GPU prefix cache, reproducing the paper
"Recency/Frequency Adaptive KV Caching for Large Language Model Serving"
(arXiv:2606.21238) and its reference implementation (the V0-era evictor in
github.com/Y-aang/vllm-ARC, vLLM v0.7.3).

Faithful-to-paper elements (classic ARC / the authors' code):
- T1/T2: resident cache split into recency (accessed once) and frequency
  (re-accessed) lists, keyed by physical block id, LRU-ordered.
- B1/B2: ghost lists that store only content hashes (never block ids).
- Adaptive target ``p`` for the T1 partition: a ghost hit in B1 grows ``p``
  by ``max(1, |B2| // |B1|)``; a ghost hit in B2 shrinks it by
  ``max(1, |B1| // |B2|)``; ``0 <= p <= c``.
- REPLACE: evict LRU of T1 (hash goes to B1) when ``|T1| > p``, otherwise
  LRU of T2 (hash goes to B2).
- Cache hit on a resident block promotes it to T2 (on its next free).
- Ghost-list bounds ``|L1| <= c`` and ``|L2| <= c`` where
  ``L1 = T1 + active-T1 + B1`` and ``L2 = T2 + active-T2 + B2``.

Engineering adaptations required by the V1 architecture (documented in the
reproduction report):
- Ghost hits are detected when a *fresh* block hash is inserted
  (``on_block_cached``), i.e. by the request that regenerates an evicted
  prefix. The authors detect them lazily at eviction time via
  ``evict(content_hash=...)``; V1 computes new block hashes after
  allocation, so insertion time is the equivalent hook. Each ghost entry is
  consumed exactly once, giving natural per-request deduplication.
- Victim selection replaces the free-queue-head (LRU) rule inside
  ``BlockPool.get_new_blocks``. Blocks referenced by any request have
  ``ref_cnt > 0`` and are never present in T1/T2, so they can never be
  evicted (refcount protection is structural).
- Uncached (hashless) free blocks are always reused before cached ones,
  preserving vLLM's LIFO locality behavior for hashless blocks; they are not
  part of ARC's key space.
- A hit is signaled exactly once per free->referenced transition
  (``BlockPool.touch`` with ``ref_cnt == 0``); refcount 1->2 (concurrent
  sharing) is not counted as an additional access.
"""

import time
from collections import OrderedDict
from typing import Optional

from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

POLICY_PROFILING = True  # overwritten by envs at BlockPool init time


class PolicyStats:
    """Cumulative control-plane counters and wall-clock time for one policy.

    Time is accumulated with ``time.perf_counter_ns()`` around the policy
    maintenance regions only: touch promotion, free insertion, fresh-hash
    classification, and victim selection (including ghost maintenance).
    Model execution and GPU waits are excluded. The default LRU path and ARC
    accumulate through this class with identical boundaries, so the reported
    overheads are comparable across the two arms.
    """

    def __init__(self, policy_name: str, profile: bool = True):
        self.policy_name = policy_name
        self.profile = profile
        # Operation counters (both arms).
        self.touch_ops = 0  # blocks promoted/re-referenced via touch (hits)
        self.free_ops = 0  # cached blocks freed back into the policy
        self.insert_ops = 0  # fresh block hashes classified by the policy
        self.evict_ops = 0  # cached blocks selected as replacement victims
        self.uncached_pops = 0  # hashless blocks reused (outside ARC key space)
        # ARC-specific counters (zero for LRU).
        self.t1_hits = 0
        self.t2_hits = 0
        self.promotions = 0  # T1 -> T2 promotions on hit
        self.ghost_b1_hits = 0
        self.ghost_b2_hits = 0
        self.evictions_t1 = 0
        self.evictions_t2 = 0
        # Cumulative policy wall time (ns), by region.
        self.touch_ns = 0
        self.free_ns = 0
        self.insert_ns = 0
        self.evict_ns = 0

    def snapshot(self) -> dict:
        return {
            "policy": self.policy_name,
            "touch_ops": self.touch_ops,
            "free_ops": self.free_ops,
            "insert_ops": self.insert_ops,
            "evict_ops": self.evict_ops,
            "uncached_pops": self.uncached_pops,
            "t1_hits": self.t1_hits,
            "t2_hits": self.t2_hits,
            "promotions": self.promotions,
            "ghost_b1_hits": self.ghost_b1_hits,
            "ghost_b2_hits": self.ghost_b2_hits,
            "evictions_t1": self.evictions_t1,
            "evictions_t2": self.evictions_t2,
            "touch_ns": self.touch_ns,
            "free_ns": self.free_ns,
            "insert_ns": self.insert_ns,
            "evict_ns": self.evict_ns,
        }


class ARCPolicy:
    """ARC replacement policy over the prefix-cached GPU blocks.

    Args:
        capacity_blocks: total number of GPU blocks in the pool (``c``),
            matching the authors' ``max_size=num_blocks``.
        stats: shared ``PolicyStats`` accumulator.
    """

    def __init__(self, capacity_blocks: int, stats: PolicyStats):
        self.c = capacity_blocks
        self.p = 0  # adaptive target size of T1 (0 <= p <= c)
        # Resident lists (ref_cnt == 0 cached blocks), LRU-first order.
        self.T1: OrderedDict[int, BlockHashWithGroupId] = OrderedDict()
        self.T2: OrderedDict[int, BlockHashWithGroupId] = OrderedDict()
        # Ghost lists: content-hash identity only, oldest-first order.
        self.B1: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()
        self.B2: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()
        # Classification of active (ref_cnt > 0) cached blocks: 1 -> T1,
        # 2 -> T2, applied when the block is freed. Equivalent to the
        # authors' T1_active/T2_active sets.
        self._pending: dict[int, int] = {}
        self._n_active_t1 = 0
        self._n_active_t2 = 0
        self.stats = stats

    # ------------------------------------------------------------------
    # Lifecycle events (called from BlockPool)
    # ------------------------------------------------------------------

    def on_block_cached(self, block_id: int, block_hash: BlockHashWithGroupId,
                        depth: int = 0, parent_hash=None):
        """A fresh (block_hash, block) mapping was just inserted for an
        active block. Classify it: ghost hit -> T2 (and adapt ``p``),
        otherwise -> T1. (depth/parent_hash are accepted for protocol
        compatibility with other GPU policies; ARC does not use them.)
        """
        target = 1
        if block_hash in self.B1:
            delta = max(1, len(self.B2) // max(1, len(self.B1)))
            self.p = min(self.p + delta, self.c)
            del self.B1[block_hash]
            self.stats.ghost_b1_hits += 1
            target = 2
        elif block_hash in self.B2:
            delta = max(1, len(self.B1) // max(1, len(self.B2)))
            self.p = max(self.p - delta, 0)
            del self.B2[block_hash]
            self.stats.ghost_b2_hits += 1
            target = 2
        self._set_pending(block_id, target)
        self._prune_ghosts()

    def on_touch(self, block_id: int, block_hash=None) -> None:
        """A cached block in the free list (ref_cnt == 0) is being
        re-referenced (prefix-cache hit). Promote it: it will re-enter T2
        when freed. Called once per free->referenced transition only.
        (block_hash accepted for protocol compatibility; unused by ARC.)
        """
        if block_id in self.T1:
            del self.T1[block_id]
            self.stats.t1_hits += 1
            self.stats.promotions += 1
            self._set_pending(block_id, 2)
        elif block_id in self.T2:
            del self.T2[block_id]
            self.stats.t2_hits += 1
            self._set_pending(block_id, 2)

    def on_free(self, block_id: int, block_hash: BlockHashWithGroupId) -> None:
        """A cached block's ref_cnt dropped to 0: insert it into its target
        list at the MRU end.
        """
        target = self._pending.pop(block_id, 1)
        if target == 1:
            self._n_active_t1 -= 1
            self.T1[block_id] = block_hash
        else:
            self._n_active_t2 -= 1
            self.T2[block_id] = block_hash
        # The authors prune ghosts at insertion time only; we additionally
        # prune here so |L1| <= c and |L2| <= c hold at every observation
        # point (the bound itself is identical; ghosts are just trimmed at
        # the same LRU-first order, possibly a bit earlier).
        self._prune_ghosts()

    def select_victim(self) -> Optional[tuple[int, BlockHashWithGroupId]]:
        """ARC REPLACE: pick the block to evict among cached free blocks.
        The evicted content hash is pushed into the matching ghost list.
        """
        if self.T1 and (len(self.T1) > self.p or not self.T2):
            block_id, block_hash = self.T1.popitem(last=False)
            self.B1[block_hash] = None
            self.stats.evictions_t1 += 1
            return block_id, block_hash
        if self.T2:
            block_id, block_hash = self.T2.popitem(last=False)
            self.B2[block_hash] = None
            self.stats.evictions_t2 += 1
            return block_id, block_hash
        return None

    def forget_block(self, block_id: int) -> None:
        """Drop a block from the policy (external eviction path / block whose
        hash was removed while free). No ghost entry is created.
        """
        if block_id in self.T1:
            del self.T1[block_id]
        elif block_id in self.T2:
            del self.T2[block_id]

    def reset(self) -> None:
        """Clear all policy state (used by ``reset_prefix_cache``)."""
        self.p = 0
        self.T1.clear()
        self.T2.clear()
        self.B1.clear()
        self.B2.clear()
        self._pending.clear()
        self._n_active_t1 = 0
        self._n_active_t2 = 0

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def lengths(self) -> dict:
        return {
            "t1": len(self.T1),
            "t2": len(self.T2),
            "b1": len(self.B1),
            "b2": len(self.B2),
            "p": self.p,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _set_pending(self, block_id: int, target: int) -> None:
        prev = self._pending.get(block_id)
        if prev == 1:
            self._n_active_t1 -= 1
        elif prev == 2:
            self._n_active_t2 -= 1
        self._pending[block_id] = target
        if target == 1:
            self._n_active_t1 += 1
        else:
            self._n_active_t2 += 1

    def _prune_ghosts(self) -> None:
        """Bound the ghost lists: |L1| <= c and |L2| <= c (authors'
        ``_prune_ghosts``; L1/L2 include the active sets).
        """
        l1 = len(self.T1) + self._n_active_t1 + len(self.B1)
        while l1 > self.c:
            self.B1.popitem(last=False)
            l1 -= 1
        l2 = len(self.T2) + self._n_active_t2 + len(self.B2)
        while l2 > self.c:
            self.B2.popitem(last=False)
            l2 -= 1


def now_ns() -> int:
    return time.perf_counter_ns()
