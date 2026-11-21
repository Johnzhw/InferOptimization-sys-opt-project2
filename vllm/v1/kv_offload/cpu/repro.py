# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-tier policies and manager for the experiment modes (Buoy §4.2/§4.3,
HotPrefix §3.3.2). Independent implementation from the papers.

Loaded out-of-tree via OffloadingSpecFactory:

  kv_connector_extra_config = {
      "spec_name": "ReproCPUOffloadingSpec",
      "spec_module_path": "vllm.v1.kv_offload.cpu.repro",
      "cpu_bytes_to_use": ...,
      "repro_mode": "<one of the nine modes>",
      "repro_extra": {...},       # same knobs as --eviction-policy-extra
      "gpu_num_blocks": 2048,     # for shared structure sizing
  }

The scheduler-side manager pulls the shared hotness tracker / admission
components from the in-process registry populated by BlockPool when the
GPU policy stack is configured (same process: scheduler).
"""

from collections import OrderedDict, deque
from typing import Optional

from typing_extensions import override

from vllm.v1.kv_offload.base import (
    OffloadKey,
    ReqContext,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

# ---------------------------------------------------------------------------
# CPU cache policies
# ---------------------------------------------------------------------------


class BuoyCPUPolicy(CachePolicy):
    """Buoy DRAM queue M (§4.2 Algorithm 1, evictM path): an ordered queue
    (head = most recent), a visited bit per key, and a FIFO ghost shared
    with the depth-aware admission (§4.3).

    evict(): from the tail; visited entries are re-inserted at the head
    with the bit reset (second chance); non-visited entries are removed
    and their chunk ids enter the ghost.
    """

    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        self.visited: set[OffloadKey] = set()
        self.ghost: deque = deque(maxlen=cache_capacity * 2)
        # stats
        self.second_chances = 0
        self.evictions = 0

    @override
    def get(self, key: OffloadKey) -> Optional[BlockStatus]:
        return self.blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        # Reverse-insertion order from the manager keeps ancestors nearer
        # the head; insert() itself appends to the tail in call order, so
        # callers insert deeper chunks first (see manager hook).
        self.blocks[key] = block

    @override
    def remove(self, key: OffloadKey) -> None:
        self.blocks.pop(key, None)
        self.visited.discard(key)

    @override
    def touch(self, keys, req_context) -> None:
        # Cache hit => visited = True (Alg. 1 line 6).
        for k in keys:
            if k in self.blocks:
                self.visited.add(k)

    @override
    def evict(self, n: int, protected) -> Optional[list]:
        out = []
        guard = 0
        while len(out) < n and self.blocks and guard < 8 * self.cache_capacity:
            guard += 1
            key, block = next(reversed(self.blocks.items()))  # tail
            if key in (protected or ()):
                # Protected (in-flight/pinned): move to head, keep.
                self.blocks.move_to_end(key, last=False)
                continue
            if key in self.visited:
                # Second chance: reset bit, reinsert at head (Alg.1 l.22-24).
                self.visited.discard(key)
                self.blocks.move_to_end(key, last=False)
                self.second_chances += 1
                continue
            # True eviction: FIFO ghost gets the chunk id (Alg.1 l.26).
            del self.blocks[key]
            self.ghost.append(key)
            out.append((key, block))
            self.evictions += 1
        return out if len(out) == n else None

    @override
    def clear(self) -> None:
        self.blocks.clear()
        self.visited.clear()
        self.ghost.clear()


class HotPrefixCPUPolicy(CachePolicy):
    """CPU tier for the hotprefix mode: LRU-ordered bookkeeping, but
    eviction victims are chosen by tracked hotness (frequency * clock,
    Eq. 6) ascending — the coldest candidates leave first (§3.3.2's
    comparison target is the least critical CPU node)."""

    def __init__(self, cache_capacity: int, tracker=None):
        super().__init__(cache_capacity)
        self.blocks: OrderedDict[OffloadKey, BlockStatus] = OrderedDict()
        self.tracker = tracker
        self.evictions = 0

    def _hotness(self, key: OffloadKey) -> int:
        if self.tracker is None:
            return 0
        return self.tracker.hotness(key)

    @override
    def get(self, key: OffloadKey) -> Optional[BlockStatus]:
        return self.blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block

    @override
    def remove(self, key: OffloadKey) -> None:
        self.blocks.pop(key, None)

    @override
    def touch(self, keys, req_context) -> None:
        for k in keys:
            if k in self.blocks:
                self.blocks.move_to_end(k)

    @override
    def evict(self, n: int, protected) -> Optional[list]:
        protected = protected or ()
        cands = [(self._hotness(k), i, k, b)
                 for i, (k, b) in enumerate(self.blocks.items())
                 if k not in protected]
        cands.sort(key=lambda t: (t[0], t[1]))
        if len(cands) < n:
            return None
        out = []
        for _, _, k, b in cands[:n]:
            del self.blocks[k]
            out.append((k, b))
            self.evictions += 1
        return out

    def coldest_hotness(self, protected=()) -> int:
        protected = protected or ()
        cands = [self._hotness(k) for k in self.blocks if k not in protected]
        return min(cands) if cands else 0

    @override
    def clear(self) -> None:
        self.blocks.clear()


# ---------------------------------------------------------------------------
# Admission gates (manager-facing adapters)
# ---------------------------------------------------------------------------


class BuoyStoreGate:
    """Depth-aware admission for the store path (§4.3), shared by the
    buoy mode; reuse events come from load hits (CPU hits)."""

    def __init__(self, admission):
        self.admission = admission  # BuoyAdmission
        self.rejected_stores = 0

    def admit_store(self, key: OffloadKey, depth: int, has_free: bool,
                    coldest_hotness_fn=None) -> bool:
        ok = self.admission.admit(key, depth)
        if not ok:
            self.rejected_stores += 1
        return ok

    def on_load_hit(self, key: OffloadKey, depth: int) -> None:
        self.admission.on_reuse(depth)


class HotPrefixStoreGate:
    """HotPrefix §3.3.2 two-stage admission for the store path."""

    def __init__(self, hp_admission, cpu_policy):
        self.hp = hp_admission      # HotPrefixAdmission
        self.cpu_policy = cpu_policy

    def admit_store(self, key: OffloadKey, depth: int, has_free: bool,
                    coldest_hotness_fn=None) -> bool:
        if has_free:
            return self.hp.admit(key, cpu_has_free_slot=True)
        fn = coldest_hotness_fn or (
            lambda: self.cpu_policy.coldest_hotness())
        return self.hp.admit(key, cpu_has_free_slot=False,
                             coldest_cpu_hotness=fn)

    def on_load_hit(self, key: OffloadKey, depth: int) -> None:
        # A CPU hit is a reuse of the node.
        self.hp.tracker.reuse(key)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ReproOffloadingManager(CPUOffloadingManager):
    """CPUOffloadingManager + admission gate + mode-specific CPU policy +
    chunk-depth tracking + key->parent chain (for promotion planning).

    The gate/policy are bound lazily: the connector's scheduler-side
    manager is created before the GPU BlockPool registers its policy
    stack (Scheduler init order), so the first use pulls the shared
    components from the in-process registry instead.

    Differences from the base manager (all gated by repro_mode; base
    behavior is preserved for None/unknown modes):
    - prepare_store filters candidates through the mode's admission gate
      BEFORE anything is allocated or transferred (rejected chunks never
      occupy CPU slots and never start D2H).
    - Chunk depth (block index in the request) is tracked per request via
      req_context state so admission sees the true prefix depth.
    - complete_load in buoy mode removes the key from the CPU queue after
      a successful reload (DRAM->HBM promotion removes from M).
    """

    LAZY_GATE_MODES = ("buoy", "hotprefix", "arc_hotprefix_admission")

    def __init__(self, *args, gate=None, repro_mode: str = "lru_tiered",
                 load_promotes_out: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = gate
        self.repro_mode = repro_mode
        self.load_promotes_out = load_promotes_out
        self._gate_bound = gate is not None
        # key -> parent key (chain order from store batches); bounded.
        self.parent_of: OrderedDict[OffloadKey, OffloadKey] = OrderedDict()
        self._parent_cap = 200_000
        # stats
        self.gate_rejected = 0

    def _lazy_bind(self) -> None:
        """Bind gate/CPU policy from the registry on first use (the GPU
        BlockPool registers its stack later in Scheduler init)."""
        self._gate_bound = True
        if self.repro_mode not in self.LAZY_GATE_MODES:
            return
        from vllm.v1.core.policy_stack import (build_cpu_admission,
                                               get_registered_stack)
        stack = get_registered_stack()
        import logging
        logging.getLogger(__name__).info(
            "ReproOffloadingManager lazy bind: mode=%s stack=%s stack.mode=%s "
            "tracker=%s admission=%s",
            self.repro_mode, stack is not None,
            getattr(stack, "mode", None),
            getattr(stack, "tracker", None) is not None,
            getattr(stack, "admission", None) is not None)
        if stack is None or stack.mode != self.repro_mode:
            return
        if self.repro_mode == "buoy":
            if stack.admission is not None and not isinstance(
                    self._policy, BuoyCPUPolicy):
                pol = BuoyCPUPolicy(self._num_blocks)
                gate = BuoyStoreGate(stack.admission)
                gate.admission._ghost = _SharedOrderedAsDeque(pol.ghost)
                self._policy = pol
                self.gate = gate
                self.load_promotes_out = True
        else:
            tracker = stack.tracker
            if tracker is None:
                return
            if self.repro_mode == "arc_hotprefix_admission":
                pol = _LRUWithHotnessView(self._num_blocks, tracker)
            else:
                pol = HotPrefixCPUPolicy(self._num_blocks, tracker)
            hp = build_cpu_admission(stack)
            if hp is not None:
                self._policy = pol
                self.gate = HotPrefixStoreGate(hp, pol)

    # -- depth tracking ---------------------------------------------------

    class _DepthState:
        def __init__(self, next_depth: int = 0):
            self.next_depth = next_depth

    def _depth_state(self, req_context: ReqContext) -> "_DepthState":
        st = req_context.get_state(self._DepthState)
        if st is None:
            st = self._DepthState()
            req_context.set_state(st)
        return st

    # -- overridden hooks ---------------------------------------------------

    @override
    def prepare_store(self, keys, req_context) -> PrepareStoreOutput | None:
        if not self._gate_bound:
            self._lazy_bind()
        keys = list(keys)
        if self.gate is not None and keys:
            depth = self._depth_state(req_context)
            base = depth.next_depth
            raw_len = len(keys)  # rejected chunks still occupy request depth
            has_free = self._get_num_free_blocks() >= len(keys)
            coldest_fn = None
            if not has_free and hasattr(self._policy, "coldest_hotness"):
                coldest_fn = self._policy.coldest_hotness
            admitted = [k for i, k in enumerate(keys)
                        if self.gate.admit_store(k, base + i, has_free,
                                                 coldest_fn)]
            self.gate_rejected += raw_len - len(admitted)
            keys = admitted
            depth.next_depth = base + raw_len
        # Track the chunk chain (parent = previous key of the same request).
        prev = None
        for k in keys:
            if prev is not None:
                self.parent_of[k] = prev
                if len(self.parent_of) > self._parent_cap:
                    self.parent_of.popitem(last=False)
            prev = k
        return super().prepare_store(keys, req_context)

    @override
    def complete_load(self, keys, req_context) -> None:
        if not self._gate_bound:
            self._lazy_bind()
        super().complete_load(keys, req_context)
        if self.gate is not None:
            depth = req_context.get_state(self._DepthState)
            d = depth.next_depth if depth else 0
            for k in keys:
                self.gate.on_load_hit(k, d)
        if self.load_promotes_out:
            # Buoy: a reloaded chunk now lives in HBM; drop it from M —
            # but only when no other in-flight readers still reference it
            # (the connector may load the same chunk for concurrent
            # requests; each completes separately).
            for k in keys:
                b = self._policy.get(k)
                if b is not None and b.ref_cnt == 0:
                    self._policy.remove(k)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


class ReproCPUOffloadingSpec(CPUOffloadingSpec):
    """CPUOffloadingSpec that builds ReproOffloadingManager with the mode's
    CPU policy and admission gate. Components shared with the GPU side
    (hotness tracker, admission) come from the in-process registry
    populated by BlockPool (scheduler process)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        extra = self.extra_config
        self.repro_mode = extra.get("repro_mode", "lru_tiered")
        self.gpu_num_blocks = int(extra.get("gpu_num_blocks", 2048))

    @override
    def get_manager(self):
        # Gate/policy binding is lazy (see ReproOffloadingManager): the
        # scheduler-side manager is created before the GPU BlockPool
        # registers its policy stack, so binding happens on first use.
        return ReproOffloadingManager(
            num_blocks=self.num_blocks,
            cache_policy="lru",
            cache_policy_module_path=None,
            enable_events=self.kv_events_config.enable_kv_cache_events,
            store_threshold=0,
            gate=None,
            repro_mode=self.repro_mode,
            load_promotes_out=False,
        )


class _SharedOrderedAsDeque:
    """Adapter: expose the policy's deque ghost as the OrderedDict-like
    container BuoyAdmission expects (membership + append + FIFO bound)."""

    def __init__(self, dq: deque):
        self._dq = dq

    def __contains__(self, k):
        return k in self._dq

    def __setitem__(self, k, v):
        self._dq.append(k)

    def __len__(self):
        return len(self._dq)

    def popitem(self, last=False):
        if self._dq:
            k = self._dq.popleft()
            return (k, None)
        raise KeyError("ghost empty")

    def clear(self):
        self._dq.clear()


class _LRUWithHotnessView(HotPrefixCPUPolicy):
    """LRU eviction order (arc_hotprefix_admission CPU tier) while still
    exposing coldest-hotness for the admission comparison."""

    def __init__(self, cache_capacity, tracker=None):
        super().__init__(cache_capacity, tracker)
        self.lru_only = True

    @override
    def evict(self, n: int, protected) -> Optional[list]:
        protected = protected or ()
        out = []
        for k in reversed(list(self.blocks.keys())):
            if len(out) == n:
                break
            if k in protected:
                continue
            out.append((k, self.blocks.pop(k)))
            self.evictions += 1
        return out if len(out) == n else None
