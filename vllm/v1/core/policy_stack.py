# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU-side policy stack for the nine experiment modes.

Components (see reproduction/mechanism_mapping.md for paper references):

- ``PrefixTreeIndex``: block-aligned logical prefix-tree bookkeeping over
  the chained block hashes: tracks, for each resident cached GPU block,
  its parent hash and the set of resident children. Provides leaf
  membership (no resident children), which HotPrefix §3.3.1/§3.4.1 uses
  to restrict eviction/promotion candidates to leaves.
- ``HotPrefixGPUPolicy``: hotness-aware eviction (HotPrefix §3.3.1,
  Eq. 5 priority = (frequency + clock) / length; length is the constant
  block size in the block-aligned adaptation, so ordering is driven by
  frequency + clock). Candidates are leaf blocks only; referenced blocks
  are structurally excluded (they never enter the candidate set).
  Selection uses a lazy heap whose entries are invalidated by tracker
  updates (reuse) and global aging.
- ``build_policy_stack``: factory translating the unified
  ``--eviction-policy`` mode name plus ``--eviction-policy-extra`` knobs
  into the component set. Modes not using a component get ``None`` —
  inactive mechanisms carry no state and no overhead.
"""

import heapq
import json
from collections import OrderedDict
from typing import Optional

from vllm.v1.core.cache_admission import BuoyAdmission
from vllm.v1.core.hotness_tracker import CuckooHotnessTracker
from vllm.v1.core.prefix_cache_policy import ARCPolicy, PolicyStats

MODES = (
    "lru",
    "arc",
    "lru_buoy_admission",
    "arc_buoy_admission",
    "lru_tiered",
    "arc_tiered",
    "hotprefix",
    "buoy",
    "arc_hotprefix_admission",
)
# Modes that require the CPU KV offload tier (group B).
TIERED_MODES = ("lru_tiered", "arc_tiered", "hotprefix", "buoy",
                "arc_hotprefix_admission")

# ---------------------------------------------------------------------------
# In-process registry: the scheduler-side offload manager pulls shared
# components (hotness tracker, admission, GPU policy) from here. Same
# process as BlockPool (the scheduler), so direct object sharing is safe.
# ---------------------------------------------------------------------------

_REGISTERED_STACK: Optional["PolicyStack"] = None


def register_stack(stack: "PolicyStack") -> None:
    global _REGISTERED_STACK
    _REGISTERED_STACK = stack


def get_registered_stack() -> Optional["PolicyStack"]:
    return _REGISTERED_STACK


def build_cpu_admission(stack: Optional["PolicyStack"]):
    """Construct (or reuse) the HotPrefix §3.3.2 admission used by the
    CPU store path for hotprefix / arc_hotprefix_admission modes."""
    if stack is None or stack.tracker is None:
        return None
    if stack.hotprefix_admission is None:
        from vllm.v1.core.hot_prefix_admission import HotPrefixAdmission
        stack.hotprefix_admission = HotPrefixAdmission(
            tracker=stack.tracker,
            freq_threshold=stack.freq_threshold,
            enabled=stack.enable_cpu_admission,
        )
    return stack.hotprefix_admission


class BuoyGPUPolicy:
    """Buoy HBM queue H (§4.2): the GPU prefix cache retains only chunks
    that have been hit (visited). New chunks are computed, optionally
    offloaded asynchronously (depth-aware admission on the store path),
    and dropped from the GPU cache on free (hash unregistered -> the
    block follows the native hashless reuse order, i.e. quick demotion).

    H eviction (on allocation pressure) pops the queue tail: the CPU
    already holds an asynchronous copy (the store path offloaded it), so
    removing the hash demotes the chunk to DRAM; if the CPU copy was
    itself evicted, the chunk is fully dropped (the M-side ghost already
    recorded the eviction).

    visited hashes are shared with the CPU manager via the registry, so
    a chunk hit on the CPU tier (reload) also marks the chunk visited
    before its data lands in HBM.
    """

    def __init__(self, stats: PolicyStats, num_blocks: int):
        self.stats = stats
        self.num_blocks = num_blocks
        # H queue: head (first item) = most recent; tail = eviction end.
        self.H: dict[int, bytes] = OrderedDict()
        # block_id -> node hash for H members (inverse map).
        self._hash_of: dict[int, bytes] = {}
        # visited node hashes (hit on GPU or reloaded from CPU).
        self.visited: OrderedDict[bytes, None] = OrderedDict()
        self._visited_cap = 4 * num_blocks
        # target of active blocks: 'H' if visited, 'M' otherwise.
        self._pending: dict[int, str] = {}
        # stats
        self.drops_on_free = 0
        self.h_evictions = 0
        self.h_hits = 0

    # -- events -------------------------------------------------------------

    def on_block_cached(self, block_id: int, block_hash, depth: int = 0,
                        parent_hash=None) -> None:
        target = "H" if block_hash in self.visited else "M"
        self._pending[block_id] = target

    def on_touch(self, block_id: int, block_hash=None) -> None:
        if block_hash is None:
            return
        self._mark_visited(block_hash)
        self._pending[block_id] = "H"
        # A referenced block must leave the H queue (it was removed from
        # the free queue by BlockPool.touch); it re-enters H on free.
        self.H.pop(block_id, None)
        self._hash_of.pop(block_id, None)
        self.h_hits += 1

    def on_free(self, block_id: int, block_hash) -> None:
        target = self._pending.pop(block_id, "M")
        if target == "H":
            self.H[block_id] = block_hash
            self._hash_of[block_id] = block_hash
            self.H.move_to_end(block_id, last=False)  # head = MRU
        else:
            # Not visited: caller (BlockPool.free_blocks) drops the hash
            # via drop_on_free(); counted here for diagnostics.
            self.drops_on_free += 1

    def drop_on_free(self, block_id: int) -> bool:
        """True if the block's hash must be unregistered on free (chunk
        never hit -> not retained in the HBM queue)."""
        return self._pending.get(block_id, "M") == "M"

    def select_victim(self):
        if not self.H:
            return None
        # Pop the tail (least recently hit end of the HBM queue).
        block_id = next(reversed(self.H))
        node_hash = self.H.pop(block_id)
        self._hash_of.pop(block_id, None)
        self.h_evictions += 1
        self.stats.evict_ops += 1
        return block_id, node_hash

    def forget_block(self, block_id: int) -> None:
        self.H.pop(block_id, None)
        self._hash_of.pop(block_id, None)
        self._pending.pop(block_id, None)

    def reset(self) -> None:
        self.H.clear()
        self._hash_of.clear()
        self.visited.clear()
        self._pending.clear()

    def _mark_visited(self, node_hash: bytes) -> None:
        self.visited[node_hash] = None
        self.visited.move_to_end(node_hash, last=False)
        while len(self.visited) > self._visited_cap:
            self.visited.popitem(last=True)

    def lengths(self) -> dict:
        return {
            "t1": len(self.H),  # HBM queue length (report slot t1)
            "t2": len(self.visited), "b1": 0, "b2": 0, "p": 0,
        }


def parse_extra(raw: Optional[str | dict]) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    return json.loads(raw)


class PrefixTreeIndex:
    """Parent/children index over resident cached GPU blocks.

    Maintained from BlockPool events; the chained block hash of a block
    identifies the prefix path root->node, and the previous block's hash
    in the same request is the parent. All operations are O(1).
    """

    def __init__(self):
        self.node_of: dict[int, bytes] = {}       # block_id -> node hash
        self.parent_hash_of: dict[int, bytes] = {}  # block_id -> parent hash
        self.block_of: dict[bytes, int] = {}      # node hash -> block_id
        self.children_of: dict[bytes, set] = {}   # parent hash -> child ids

    def on_cached(self, block_id: int, node_hash: bytes, parent_hash):
        self.node_of[block_id] = node_hash
        self.block_of[node_hash] = block_id
        if parent_hash is not None:
            self.parent_hash_of[block_id] = parent_hash
            self.children_of.setdefault(parent_hash, set()).add(block_id)

    def on_evicted(self, block_id: int, node_hash: bytes):
        self.node_of.pop(block_id, None)
        self.block_of.pop(node_hash, None)
        parent = self.parent_hash_of.pop(block_id, None)
        if parent is not None:
            kids = self.children_of.get(parent)
            if kids is not None:
                kids.discard(block_id)

    def is_leaf_block(self, block_id: int) -> bool:
        node = self.node_of.get(block_id)
        if node is None:
            return True
        return not self.children_of.get(node)

    def clear(self):
        self.node_of.clear()
        self.parent_hash_of.clear()
        self.block_of.clear()
        self.children_of.clear()


class HotPrefixGPUPolicy:
    """Hotness-aware GPU eviction (HotPrefix §3.3.1, Algorithm 1).

    Event API mirrors ARCPolicy so BlockPool can drive both identically.
    Depth/parent come from the caching path (chain order).
    """

    def __init__(self, tracker: CuckooHotnessTracker, stats: PolicyStats,
                 num_blocks: int, block_tokens: int = 16):
        self.tracker = tracker
        self.stats = stats
        self.num_blocks = num_blocks
        self.block_tokens = block_tokens
        self.tree = PrefixTreeIndex()
        # leaf candidates: block_id -> node hash (resident, free, cached)
        self._leaves: dict[int, bytes] = {}
        # lazy min-heap of (priority, block_id)
        self._heap: list = []
        self._heap_epoch = 0
        self._heap_dirty = False

    # ------------------------------------------------------------------
    # Events (driven by BlockPool, same protocol as ARCPolicy)
    # ------------------------------------------------------------------

    def on_block_cached(self, block_id: int, block_hash, depth: int = 0,
                        parent_hash=None) -> None:
        self.tracker.insert(block_hash, depth)
        # My parent is no longer a leaf.
        if parent_hash is not None:
            pb = self.tree.block_of.get(parent_hash)
            if pb is not None:
                self._leaves.pop(pb, None)
        self.tree.on_cached(block_id, block_hash, parent_hash)

    def on_touch(self, block_id: int, block_hash=None) -> None:
        if block_hash is None:
            return
        self.tracker.reuse(block_hash)
        # A referenced block is no longer an eviction candidate (it was
        # removed from the free queue by BlockPool.touch); drop it from
        # the candidate set. It re-enters on its next free if still a leaf.
        self._leaves.pop(block_id, None)

    def on_free(self, block_id: int, block_hash) -> None:
        # Becomes an eviction candidate iff it is a leaf.
        if self.tree.is_leaf_block(block_id):
            self._leaves[block_id] = block_hash
            self._push(block_id)

    def select_victim(self):
        """Pop the lowest-priority leaf (Eq. 5; constant length)."""
        if not self._leaves:
            return None
        self._ensure_heap()
        while self._heap:
            prio, block_id = heapq.heappop(self._heap)
            node_hash = self._leaves.get(block_id)
            if node_hash is None:
                continue  # stale entry
            cur = self._priority(node_hash)
            if cur != prio:
                # stale priority: refresh and retry
                heapq.heappush(self._heap, (cur, block_id))
                if self._heap and self._heap[0][1] == block_id:
                    prio, block_id = heapq.heappop(self._heap)
                    node_hash = self._leaves.get(block_id)
                    if node_hash is None:
                        continue
                else:
                    continue
            del self._leaves[block_id]
            self.tree.on_evicted(block_id, node_hash)
            self.stats.evict_ops += 1
            return block_id, node_hash
        return None

    def forget_block(self, block_id: int) -> None:
        node = self._leaves.pop(block_id, None)
        if node is not None:
            self.tree.on_evicted(block_id, node)

    def reset(self) -> None:
        self._leaves.clear()
        self._heap.clear()
        self.tree.clear()
        self._heap_epoch += 1

    # ------------------------------------------------------------------

    def _priority(self, node_hash) -> float:
        rec = self.tracker.get(node_hash)
        if rec is None:
            return 0.0
        freq, clock, _ = rec
        # Eq.5 with constant length(block_tokens): the division by a
        # constant preserves ordering; use float (not floor) so the
        # frequency term keeps full resolution at block granularity.
        return (freq + clock) / self.block_tokens

    def _push(self, block_id: int) -> None:
        node_hash = self._leaves.get(block_id)
        if node_hash is not None:
            heapq.heappush(self._heap, (self._priority(node_hash), block_id))

    def _ensure_heap(self) -> None:
        if len(self._heap) > 8 * max(1, len(self._leaves)) + 64:
            self._heap = [(self._priority(h), b) for b, h in self._leaves.items()]
            heapq.heapify(self._heap)

    def on_aged(self) -> None:
        """Called after tracker.age(): all priorities changed globally."""
        self._heap = [(self._priority(h), b) for b, h in self._leaves.items()]
        heapq.heapify(self._heap)

    def lengths(self) -> dict:
        return {
            "t1": len(self._leaves),  # leaf candidates (report as t1 slot)
            "t2": 0, "b1": 0, "b2": 0, "p": 0,
        }


class PolicyStack:
    """Component set for one experiment mode.

    Fields are None when the mode does not use the mechanism; unused
    mechanisms must have no state, no timers, no background work.
    """

    def __init__(self, mode: str, extra: dict):
        self.mode = mode
        self.extra = extra
        self.gpu_policy = None            # ARCPolicy | HotPrefixGPUPolicy
        self.admission = None             # BuoyAdmission (GPU-side gate)
        self.tracker = None               # CuckooHotnessTracker
        self.hotprefix_admission = None   # HotPrefixAdmission (CPU-side)
        self.gpu_eviction_mode = "lru"    # lru | arc | hotprefix
        self.needs_offload = mode in TIERED_MODES
        # knobs (paper defaults unless overridden)
        self.max_age = int(extra.get("max_age", 255))
        self.age_interval = int(extra.get("age_interval", 256))
        self.freq_threshold = int(extra.get("freq_threshold", 10))
        self.enable_promotion = bool(extra.get("promotion", True))
        self.enable_cpu_admission = bool(extra.get("cpu_admission", True))
        self.enable_depth_admission = bool(extra.get("depth_admission", True))
        self.fp_bits = int(extra.get("fp_bits", 16))

    def snapshot(self) -> dict:
        out = {"mode": self.mode}
        if self.admission is not None:
            out["buoy_admission"] = self.admission.snapshot()
        if self.hotprefix_admission is not None:
            out["hp_admission"] = self.hotprefix_admission.snapshot()
        if self.tracker is not None:
            out["tracker_entries"] = len(self.tracker)
            out["tracker_aging_calls"] = self.tracker.aging_calls
            out["tracker_aging_ms"] = round(
                getattr(self.tracker, "_aging_ops_ns", 0) / 1e6, 2)
            out["tracker_kicks"] = getattr(self.tracker, "kicks", 0)
            out["tracker_dropped"] = getattr(self.tracker, "dropped", 0)
        return out


def build_policy_stack(eviction_policy: str, extra_raw, num_blocks: int,
                       stats: PolicyStats) -> PolicyStack:
    """Build the component stack for a mode. GPU BlockPool integration
    only; offload-side components (CPU policies, promotion planner) are
    attached later by the offload integration layer."""
    extra = parse_extra(extra_raw)
    if eviction_policy not in MODES:
        raise ValueError(
            f"Unknown eviction policy {eviction_policy!r}; expected one of {MODES}")
    stack = PolicyStack(eviction_policy, extra)

    if eviction_policy == "arc":
        stack.gpu_policy = ARCPolicy(num_blocks, stats)
        stack.gpu_eviction_mode = "arc"
    elif eviction_policy in ("arc_buoy_admission", "arc_tiered",
                             "arc_hotprefix_admission"):
        stack.gpu_policy = ARCPolicy(num_blocks, stats)
        stack.gpu_eviction_mode = "arc"

    if eviction_policy in ("lru_buoy_admission", "arc_buoy_admission", "buoy"):
        stack.admission = BuoyAdmission(
            ghost_capacity=num_blocks * 2,
            enabled=stack.enable_depth_admission,
        )
        if eviction_policy == "buoy":
            stack.gpu_policy = BuoyGPUPolicy(stats, num_blocks)
            stack.gpu_eviction_mode = "buoy_h"

    if eviction_policy in ("hotprefix", "arc_hotprefix_admission"):
        buckets = max(1024, 4 * num_blocks)
        # keep power of two
        buckets = 1 << (buckets - 1).bit_length()
        stack.tracker = CuckooHotnessTracker(
            num_buckets=buckets, max_age=stack.max_age,
            fp_bits=stack.fp_bits)
        if eviction_policy == "hotprefix":
            stack.gpu_policy = HotPrefixGPUPolicy(
                stack.tracker, stats, num_blocks)
            stack.gpu_eviction_mode = "hotprefix"

    return stack
