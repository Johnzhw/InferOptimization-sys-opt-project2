# SPDX-License-Identifier: Apache-2.0
# EWSJF request queue: the tactical scheduling loop (paper Section 3.2,
# Algorithm 1) realized as a RequestQueue whose pop/peek order follows
# Density-Weighted Scoring across performance-homogeneous prompt-length
# queues. The strategic loop (Section 3.1: Monitor + Optimizer running
# Refine-and-Prune and Bayesian meta-optimization) runs in a daemon thread
# and swaps immutable policy snapshots atomically.
#
# Thread-safety contract: the scheduler event loop is the only mutator of
# the partition deques. The strategic thread only reads monitor snapshots
# (lock-protected ring buffer) and replaces `self._policy` with a single
# reference assignment (atomic under the GIL). Re-partitioning happens on
# the scheduler thread, triggered through a flag checked inside
# add/pop (cheap, keeps deque mutation single-threaded).

from __future__ import annotations

import bisect
import logging
import math
import os
import threading
import time
from collections import deque
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.v1.core.sched.ewsjf_policy import (
    EWSJFMetaOptimizer,
    EWSJFMonitor,
    EWSJFPolicy,
    bubble_route,
    calculate_score,
    refine_and_prune,
)
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.config import SchedulerConfig

logger = logging.getLogger(__name__)

# Shared-policy registry: one SchedulerConfig creates two EWSJF queues
# (self.waiting and self.skipped_waiting). They must share a single policy
# and a single strategic loop so routing and scoring stay consistent and
# only one background thread exists per engine.
_SHARED: dict[int, "EWSJFRequestQueue"] = {}


def _clear_shared() -> None:
    """Test helper: drop registry entries (engines are long-lived)."""
    _SHARED.clear()


class _Partition:
    """One prompt-length queue: contiguous [min_len, max_len) plus FIFO."""

    __slots__ = ("min_len", "max_len", "reqs", "is_bubble",
                 "n_assigned", "sum_assigned_len")

    def __init__(self, min_len: int, max_len: int, is_bubble: bool = False):
        self.min_len = min_len
        self.max_len = max_len
        self.reqs: deque[Request] = deque()
        self.is_bubble = is_bubble
        self.n_assigned = 0
        self.sum_assigned_len = 0

    def mean_len(self) -> float:
        if self.n_assigned:
            return self.sum_assigned_len / self.n_assigned
        return (self.min_len + self.max_len) / 2.0

    def __len__(self) -> int:
        return len(self.reqs)


class EWSJFRequestQueue(RequestQueue):
    """EWSJF tactical loop (Algorithm 1) as a drop-in RequestQueue.

    Scheduling parameters are read from SchedulerConfig EWSJF fields; all
    have safe defaults so a bare `--scheduling-policy ewsjf` works. With
    scheduling-policy != ewsjf this class is never instantiated (zero
    state, zero overhead on FCFS/priority configurations).
    """

    def __init__(self, scheduler_config: SchedulerConfig | None = None):
        cfg = scheduler_config
        self.max_queues: int = getattr(cfg, "ewsjf_max_queues", 32) if cfg else 32
        self.alpha: float = getattr(cfg, "ewsjf_alpha", 2.0) if cfg else 2.0
        self.window: int = getattr(cfg, "ewsjf_window", 4096) if cfg else 4096
        self.strategic_interval: float = (
            getattr(cfg, "ewsjf_strategic_interval", 30.0) if cfg else 30.0
        )
        self.bubble_width: int = (
            getattr(cfg, "ewsjf_bubble_width", 256) if cfg else 256
        )
        self.meta_trials: int = getattr(cfg, "ewsjf_meta_trials", 0) if cfg else 0
        self._meta_disabled = os.environ.get("VLLM_EWSJF_DISABLE_META", "0") == "1"

        self.monitor = EWSJFMonitor(capacity=self.window)
        self._shared_primary: "EWSJFRequestQueue | None" = None
        self._policy = EWSJFPolicy(boundaries=(256, 1024, 4096), alpha=self.alpha)
        self._partitions: list[_Partition] = self._partitions_from_policy(
            self.policy.boundaries
        )
        # peek cache: (partition index, head request) so consecutive
        # peek->pop pairs do not rescore.
        self._peeked: tuple[int, Request] | None = None
        # Mechanism telemetry + management overhead accounting.
        self._counters = {
            "adds": 0, "pops": 0, "peeks": 0, "scores": 0,
            "bubble_created": 0, "repartitions": 0, "meta_trials_done": 0,
            "tactical_time_us": 0.0,
        }
        self._repartition_flag = False
        self._stop = threading.Event()
        self._strategic: threading.Thread | None = None
        # Secondary queues (skipped_waiting) share the primary's policy,
        # monitor and strategic thread: registered per SchedulerConfig.
        primary = _SHARED.get(id(cfg)) if cfg is not None else None
        if primary is not None:
            self._policy = primary._policy
            self.monitor = primary.monitor
            self._shared_primary = primary
            logger.info_once(
                "EWSJF scheduler secondary queue attached to primary policy"
            )
        else:
            self._shared_primary = None
            if cfg is not None:
                _SHARED[id(cfg)] = self
            if self.strategic_interval > 0:
                self._strategic = threading.Thread(
                    target=self._strategic_loop, daemon=True,
                    name="ewsjf-strategic",
                )
                self._strategic.start()
        logger.info(
            "EWSJF scheduler active: max_queues=%d alpha=%.2f window=%d "
            "strategic_interval=%.1fs meta_trials=%d boundaries=%s",
            self.max_queues, self.alpha, self.window, self.strategic_interval,
            self.meta_trials, self.policy.boundaries,
        )

    # ------------------------------------------------------------------
    # policy access: secondary queues follow the primary's live policy
    # ------------------------------------------------------------------

    @property
    def policy(self) -> EWSJFPolicy:
        if self._shared_primary is not None:
            return self._shared_primary._policy
        return self._policy

    # ------------------------------------------------------------------
    # partition maintenance
    # ------------------------------------------------------------------

    def _partitions_from_policy(self, boundaries: tuple[int, ...]) -> list[_Partition]:
        parts = []
        low = 0
        for up in boundaries:
            parts.append(_Partition(low, up))
            low = up
        parts.append(_Partition(low, self.policy.max_len))
        return parts

    def _route(self, request: Request) -> int:
        """Dispatcher: bubble-aware routing into a partition index."""
        plen = self._prompt_len(request)
        action, q, nmin, nmax = bubble_route(
            plen, self.policy.boundaries, self.bubble_width
        )
        if action == "assign":
            return min(q, len(self._partitions) - 1)
        # true gap detected: create bubble partition at position q
        self._partitions.insert(q, _Partition(nmin, nmax, is_bubble=True))
        self._counters["bubble_created"] += 1
        return q

    @staticmethod
    def _prompt_len(request: Request) -> int:
        pt = request.prompt_token_ids
        return max(1, len(pt)) if pt else 1

    # ------------------------------------------------------------------
    # tactical loop (Algorithm 1): score heads, argmax, greedy fill
    # ------------------------------------------------------------------

    def _pick(self) -> int:
        """ArgMax over non-empty queue head scores (Algorithm 1 lines 3-18)."""
        t0 = time.perf_counter()
        best_q, best_s = -1, -1.0
        for q, part in enumerate(self._partitions):
            if part.reqs:
                req = part.reqs[0]
                s = calculate_score(
                    prompt_len=self._prompt_len(req),
                    wait_time=max(0.0, time.time() - req.arrival_time),
                    queue_index=q,
                    queue_mean_len=part.mean_len(),
                    policy=self.policy,
                )
                self._counters["scores"] += 1
                if s > best_s:
                    best_q, best_s = q, s
        self._counters["tactical_time_us"] += (time.perf_counter() - t0) * 1e6
        return best_q

    # ------------------------------------------------------------------
    # RequestQueue interface
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        self._apply_new_policy_if_pending()
        q = self._route(request)
        part = self._partitions[q]
        part.reqs.append(request)
        part.n_assigned += 1
        part.sum_assigned_len += self._prompt_len(request)
        self._peeked = None
        self._counters["adds"] += 1

    def pop_request(self) -> Request:
        self._apply_new_policy_if_pending()
        q = self._pick()
        part = self._partitions[q]
        req = part.reqs.popleft()
        self.monitor.observe(
            req.request_id, self._prompt_len(req),
            max(0.0, time.time() - req.arrival_time),
        )
        self._peeked = None
        self._counters["pops"] += 1
        return req

    def peek_request(self) -> Request:
        if self._peeked is not None:
            q0, req0 = self._peeked
            part0 = self._partitions[q0] if q0 < len(self._partitions) else None
            if part0 is not None and part0.reqs and part0.reqs[0] is req0:
                self._counters["peeks"] += 1
                return req0
        q = self._pick()
        req = self._partitions[q].reqs[0]
        self._peeked = (q, req)
        self._counters["peeks"] += 1
        return req

    def prepend_request(self, request: Request) -> None:
        """Preempted request returns to the head of its own partition
        (partition FIFO order preserved across preemption)."""
        self._apply_new_policy_if_pending()
        q = self._route(request)
        self._partitions[q].reqs.appendleft(request)
        self._peeked = None

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        for part in self._partitions:
            try:
                part.reqs.remove(request)
                self._peeked = None
                return
            except ValueError:
                continue

    def remove_requests(self, requests: Iterable[Request]) -> None:
        gone = {id(r) for r in requests}
        for part in self._partitions:
            if part.reqs:
                keep = deque(r for r in part.reqs if id(r) not in gone)
                if len(keep) != len(part.reqs):
                    part.reqs = keep
                    self._peeked = None

    def __bool__(self) -> bool:
        return any(p.reqs for p in self._partitions)

    def __len__(self) -> int:
        return sum(len(p.reqs) for p in self._partitions)

    def __iter__(self):
        """Yield every pending request exactly once: partitions ordered by
        their current head score (descending), FIFO within a partition.
        The first element always matches the tactical pop order."""
        scored = []
        for q, part in enumerate(self._partitions):
            if part.reqs:
                req = part.reqs[0]
                s = calculate_score(
                    prompt_len=self._prompt_len(req),
                    wait_time=max(0.0, time.time() - req.arrival_time),
                    queue_index=q,
                    queue_mean_len=part.mean_len(),
                    policy=self.policy,
                )
                scored.append((-s, part.reqs[0].arrival_time, part))
        scored.sort(key=lambda t: (t[0], t[1]))
        out: list[Request] = []
        for _, _, part in scored:
            out.extend(part.reqs)
        return iter(out)

    # ------------------------------------------------------------------
    # strategic loop (Section 3.1): Monitor -> Optimizer -> new policy
    # ------------------------------------------------------------------

    def _apply_new_policy_if_pending(self) -> None:
        if not self._repartition_flag:
            return
        self._repartition_flag = False
        pending = [r for p in self._partitions for r in p.reqs]
        boundaries = self.policy.boundaries
        parts = self._partitions_from_policy(boundaries)
        for req in pending:
            q = min(bisect.bisect_right(boundaries, self._prompt_len(req)),
                    len(parts) - 1)
            parts[q].reqs.append(req)
        self._partitions = parts
        self._peeked = None
        self._counters["repartitions"] += 1

    def _strategic_loop(self) -> None:
        meta = EWSJFMetaOptimizer()
        trial_theta: tuple[float, ...] | None = None
        trials_done = 0
        tick = 0
        while not self._stop.is_set():
            if self._stop.wait(self.strategic_interval):
                break
            tick += 1
            try:
                lengths = self.monitor.snapshot_lengths()
                obs = self.monitor.snapshot()
                if not lengths:
                    continue
                # offline (history) mode: full Refine-and-Prune on window
                boundaries = refine_and_prune(
                    lengths, max_queues=self.max_queues, alpha=self.alpha
                )
                # online (real-time) mode: Bayesian meta-optimization of the
                # scoring weights (Section 4.4.2, Eq.(5)) while trial budget
                # lasts; frozen weights afterwards.
                meta_on = (
                    self.meta_trials > 0
                    and not self._meta_disabled
                    and trials_done < self.meta_trials
                )
                if meta_on:
                    if trial_theta is None:
                        trial_theta = meta.next_theta()
                    stats = self._window_stats(obs, boundaries)
                    meta.record(trial_theta,
                                EWSJFMetaOptimizer.reward_from_stats(stats),
                                stats)
                    trials_done += 1
                    a_urg, b_urg, b_fair, alpha_t = trial_theta
                    self._policy = EWSJFPolicy(
                        boundaries=boundaries,
                        a_urg=a_urg, b_urg=b_urg, b_fair=b_fair, alpha=alpha_t,
                    )
                    trial_theta = meta.next_theta()
                else:
                    self._policy = EWSJFPolicy(boundaries=boundaries)
                self._repartition_flag = True
                self._counters["meta_trials_done"] = trials_done
                if tick % 4 == 1:
                    logger.info(
                        "EWSJF strategic tick %d: queues=%d observed=%d "
                        "boundaries=%s... meta_trials=%d",
                        tick, len(boundaries), len(lengths),
                        boundaries[:8], trials_done,
                    )
                    c = self.counters()
                    logger.info(
                        "EWSJF tactical telemetry: pops=%d scores=%d "
                        "tactical_us_per_pop=%.2f bubbles=%d repartitions=%d",
                        c["pops"], c["scores"], c["tactical_us_per_pop"],
                        c["bubble_created"], c["repartitions"],
                    )
            except Exception:  # strategic loop must never kill serving
                logger.exception("EWSJF strategic loop iteration failed")

    @staticmethod
    def _window_stats(obs, boundaries) -> dict:
        """Eq.(5) component statistics from one monitor window."""
        n_q = max(len(boundaries), 1)
        counts = [0] * (n_q + 1)
        per_q: list[list[int]] = [[] for _ in range(n_q + 1)]
        waits = [w for _, _, w in obs]
        for _, plen, _w in obs:
            qi = bisect.bisect_right(boundaries, plen)
            counts[qi] += 1
            per_q[qi].append(plen)
        spreads = [
            (max(lst) - min(lst)) / 4096.0 for lst in per_q if lst
        ]
        mean_c = sum(counts) / len(counts)
        occ_cv = (
            math.sqrt(sum((c - mean_c) ** 2 for c in counts) / len(counts))
            / mean_c
            if mean_c > 0
            else 0.0
        )
        return {
            "mean_within_spread": sum(spreads) / len(spreads) if spreads else 0.0,
            "occupancy_cv": occ_cv,
            "queue_count_ratio": len([c for c in counts if c]) / max(1, n_q),
            "mean_wait_s": sum(waits) / len(waits) if waits else 0.0,
        }

    # ------------------------------------------------------------------
    # telemetry
    # ------------------------------------------------------------------

    def counters(self) -> dict:
        c = dict(self._counters)
        pops = max(c["pops"], 1)
        c["tactical_us_per_pop"] = c["tactical_time_us"] / pops
        c["active_queues"] = sum(1 for p in self._partitions if p.reqs)
        c["total_partitions"] = len(self._partitions)
        return c
