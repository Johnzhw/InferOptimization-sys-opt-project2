# SPDX-License-Identifier: Apache-2.0
# EWSJF (Effective Workload-based Shortest Job First) adaptive scheduler.
# Independent implementation from the paper's description:
# "EWSJF: An Adaptive Scheduler with Hybrid Partitioning for Mixed-Workload
#  LLM Inference" (KDD'26 submission, arXiv:2601.21758).
#  - Refine-and-Prune partitioning ........ Section 4.2, Eq.(2)-(3)
#  - Density-Weighted Scoring ............. Section 4.1 Eq.(1) / Section 4.4.1 Eq.(4)
#  - On-Demand Bubble Queue Creation ...... Section 4.3 / Appendix D Algorithm 2
#  - Bayesian meta-optimization ........... Section 4.4.2 Eq.(5)
# This module is pure logic (no vLLM engine state) so every piece is unit
# testable with deterministic inputs.

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Policy value object: queue boundaries + scoring weights. Replaced atomically
# by the strategic loop; the tactical loop only ever reads one snapshot.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EWSJFPolicy:
    """Immutable scheduling policy snapshot.

    boundaries: sorted prompt-length upper bounds (exclusive) of the queues;
        queue i holds requests with prompt length in
        [boundaries[i-1], boundaries[i]) (boundaries[-1] treated as 0).
    weights: scoring weights. ``urg``/``fair`` slopes/intercepts implement the
        paper's context-aware meta-policy w(b̄_q) = a·b̄_q + b (Section 4.4.1).
    """

    boundaries: tuple[int, ...]
    w_base: float = 1.0
    a_urg: float = 0.0
    b_urg: float = 1.0
    a_fair: float = 0.0
    b_fair: float = 0.05
    alpha: float = 2.0  # gap significance ratio of Eq.(2)
    max_len: int = 1 << 17  # prompt lengths above the last boundary clamp here

    def queue_index(self, prompt_len: int) -> int:
        """Deterministic routing r -> q_i (Section 5 correctness)."""
        lo, hi = 0, len(self.boundaries)
        while lo < hi:
            mid = (lo + hi) // 2
            if prompt_len < self.boundaries[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def queue_mean_len(self, q_index: int, counts: dict[int, tuple[int, int]]) -> float:
        """Mean prompt length of queue q given {q: (n, sum_len)} statistics."""
        n, s = counts.get(q_index, (0, 0))
        return (s / n) if n else self._mid_boundary(q_index)

    def _mid_boundary(self, q_index: int) -> float:
        low = self.boundaries[q_index - 1] if q_index > 0 else 0
        high = self.boundaries[q_index] if q_index < len(self.boundaries) else self.max_len
        return (low + high) / 2.0

    def weights_for(self, mean_len: float) -> tuple[float, float, float]:
        """Context-aware weights (paper example: w_urg(b̄_q)=a_u·b̄_q+b_u)."""
        w_urg = self.a_urg * mean_len + self.b_urg
        w_fair = self.a_fair * mean_len + self.b_fair
        return self.w_base, max(w_urg, 0.0), max(w_fair, 0.0)


def prefill_cost(b: int, scale: int = 4096) -> float:
    """Estimated prefill compute cost C_prefill(b) for input length b.

    Engineering assumption (paper leaves C_prefill unspecified): a linear
    term plus a quadratic term normalized by the typical long-prompt scale,
    C_prefill(b) = b + b^2/scale. Sensitivity of the scheduler to this form
    is checked with pure-linear and pure-quadratic variants in the report.
    """
    return float(b) + (float(b) * float(b)) / float(scale)


def calculate_score(
    prompt_len: int,
    wait_time: float,
    queue_index: int,
    queue_mean_len: float,
    policy: EWSJFPolicy,
    now: float | None = None,
) -> float:
    """Density-Weighted Scoring, paper Eq.(1)/Eq.(4):

        Score(r,q) = q_f · (w_base + w_urg·c_s + w_fair·log(b+1))
        c_s = W_t / C_prefill(b)      (compute-normalized urgency)
        q_f = q_i / (b + 1)           (SJF-inspired queue factor)

    ``wait_time`` is seconds; ``prompt_len`` b is in tokens; ``queue_index``
    q_i is the 1-based queue ordinal divided by (b+1) as in the paper.
    """
    b = prompt_len
    w_base, w_urg, w_fair = policy.weights_for(queue_mean_len)
    c_s = wait_time / prefill_cost(b) if b > 0 else wait_time
    q_f = (queue_index + 1) / (b + 1)
    return q_f * (w_base + w_urg * c_s + w_fair * math.log(b + 1))


# ---------------------------------------------------------------------------
# Refine-and-Prune partitioning (Section 4.2), deterministic.
# ---------------------------------------------------------------------------


def _kmeans1d(values: list[float], k: int, iters: int = 50) -> list[int]:
    """Deterministic 1-D k-means: returns label per value.

    Initialization by equal-frequency quantiles of the sorted data, Lloyd
    iterations until labels stabilize. With k=3 this mirrors the paper's
    coarse short/medium/long anchors (Section 4.2 Stage 1).
    """
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    centers = [values[order[min(n - 1, (j * n) // k)]] for j in range(k)]
    labels = [0] * n
    for _ in range(iters):
        new_centers: list[float] = []
        # assign
        changed = False
        counts = [0] * k
        sums = [0.0] * k
        for i, v in enumerate(values):
            j = min(range(k), key=lambda c: (abs(centers[c] - v), c))
            if labels[i] != j:
                changed = True
                labels[i] = j
            counts[j] += 1
            sums[j] += v
        # update (keep empty clusters' centers)
        new_centers = [
            sums[c] / counts[c] if counts[c] else centers[c] for c in range(k)
        ]
        centers = new_centers
        if not changed:
            break
    return labels


def refine_and_prune(
    lengths: list[int],
    max_queues: int = 32,
    alpha: float = 2.0,
    min_width: int = 32,
) -> tuple[int, ...]:
    """Three-stage Refine-and-Prune over observed prompt lengths.

    Stage 1: coarse k-means (k=3) on lengths.
    Stage 2: recursive refinement — split cluster C at gap j when
             Gap_j > alpha * mean(G)   (Eq.(2)), until no significant gap
             remains or cluster width < min_width.
    Stage 3: merge adjacent queues by scheduling utility
             U(q_i,q_{i+1}) = (rho_i + rho_{i+1}) / (|b̄_{i+1}-b̄_i| + eps)
             (Eq.(3)) until <= max_queues remain.

    Returns sorted exclusive upper bounds of the surviving queues.
    Deterministic: same input list -> same output.
    """
    if not lengths:
        return (256, 1024, 4096)  # neutral default boundaries
    vals = [float(x) for x in lengths]
    labels = _kmeans1d(vals, k=3)
    clusters: list[list[float]] = [[], [], []]
    for v, lab in zip(vals, labels):
        clusters[lab].append(v)
    clusters = [sorted(c) for c in clusters if c]
    clusters.sort(key=lambda c: c[0])

    # Stage 2: recursive gap split.
    def split(c: list[float]) -> list[list[float]]:
        if len(c) < 2 or (c[-1] - c[0]) < min_width:
            return [c]
        gaps = [c[i + 1] - c[i] for i in range(len(c) - 1)]
        gmean = sum(gaps) / len(gaps)
        best_j, best_gap = -1, -1.0
        for j, g in enumerate(gaps):
            if g > alpha * gmean and g > best_gap:
                best_j, best_gap = j, g
        if best_j < 0:
            return [c]
        return split(c[: best_j + 1]) + split(c[best_j + 1 :])

    parts: list[list[float]] = []
    for c in clusters:
        parts.extend(split(c))
    parts.sort(key=lambda c: c[0])

    # Stage 3: utility-based merging (Eq.(3)) down to max_queues.
    def rho(c: list[float]) -> float:
        width = max(c[-1] - c[0], 1.0)
        return len(c) / width

    while len(parts) > max_queues:
        # merge the adjacent pair with lowest utility
        best_u, best_i = None, 0
        for i in range(len(parts) - 1):
            a, b = parts[i], parts[i + 1]
            mean_a = sum(a) / len(a)
            mean_b = sum(b) / len(b)
            u = (rho(a) + rho(b)) / (abs(mean_b - mean_a) + 1e-6)
            if best_u is None or u < best_u:
                best_u, best_i = u, i
        merged = sorted(parts[best_i] + parts[best_i + 1])
        parts = parts[:best_i] + [merged] + parts[best_i + 2 :]

    bounds = []
    for c in parts:
        up = int(max(c)) + 1
        if not bounds or up > bounds[-1]:
            bounds.append(up)
    return tuple(bounds)


# ---------------------------------------------------------------------------
# On-Demand Bubble Queue Creation (Section 4.3 / Appendix D Algorithm 2).
# ---------------------------------------------------------------------------


def bubble_route(
    prompt_len: int, boundaries: tuple[int, ...], default_bubble_width: int = 256
) -> tuple[str, int, int, int]:
    """Route a possibly gap-falling request per Appendix D Algorithm 2.

    Q_i is the last queue whose max_len <= L (adjacent lower neighbor),
    Q_{i+1} the queue above. Algorithm 2:
      1: L <= Q_i.max_len * 1.10        -> assign to Q_i
      2: L >= Q_{i+1}.min_len * 0.90    -> assign to Q_{i+1}
      3: true gap                       -> bubble queue of width
         min(default_bubble_width, Q_{i+1}.min_len - Q_i.max_len),
         centered on L and clamped to the neighbors.
    With contiguous partitions rule 2 covers in-range lengths; true gaps
    arise above the top boundary or between non-contiguous bounds (e.g.
    after a bubble insertion). Returns (action, q_index, new_min, new_max).
    """
    import bisect

    n = len(boundaries)
    i = bisect.bisect_right(boundaries, prompt_len) - 1  # Q_i index, -1 below first
    qmax_i = boundaries[i] if i >= 0 else 0
    if i >= 0 and prompt_len <= qmax_i * 1.10:
        return ("assign", i, 0, 0)
    if i + 1 < n:
        qmin_next = boundaries[i] if i >= 0 else 0  # min_len of Q_{i+1}
        if prompt_len >= qmin_next * 0.90:
            return ("assign", i + 1, 0, 0)
        # interior true gap (non-contiguous bounds only)
        lo, hi = qmax_i, boundaries[i + 1]
        rng = min(default_bubble_width, hi - lo)
        new_min = max(prompt_len - rng // 2, lo)
        new_max = min(prompt_len + rng // 2, hi)
        return ("bubble", i + 1, new_min, new_max)
    # beyond the last queue: rule 1 with 10% slack, else bubble on top
    # (gap is unbounded there, so the bubble is centered on L, clamped low)
    if i >= 0 and prompt_len <= boundaries[i] * 1.10:
        return ("assign", i, 0, 0)
    base = boundaries[-1] if n else 0
    new_min = max(prompt_len - default_bubble_width // 2, base)
    return ("bubble", n, new_min, new_min + default_bubble_width)


# ---------------------------------------------------------------------------
# Bayesian meta-optimization (Section 4.4.2, Eq.(5)), lightweight GP-UCB.
# Pure numpy-free implementation (small matrices solved directly) so no new
# dependency is introduced.
# ---------------------------------------------------------------------------


@dataclass
class MetaTrial:
    theta: tuple[float, ...]
    reward: float | None = None
    stats: dict = field(default_factory=dict)


class EWSJFMetaOptimizer:
    """GP-UCB bandit over Θ = (a_urg, b_urg, b_fair, alpha).

    a_urg is the context slope of the paper's meta-policy example
    w_urg(b̄_q) = a_u·b̄_q + b_u (Section 4.4.1): larger urgency weight for
    long-prompt queues shortens the starvation horizon of long requests.

    Reward R(Θ) = λ1·C + λ2·L − λ3·S − λ4·U follows Eq.(5):
      C compactness  : negative mean within-queue length spread (normalized)
      L load balance : negative coefficient of variation of queue occupancy
      S structure    : queue proliferation penalty
      U user penalty : mean wait time of scheduled requests
    Convergence budget 5-8 trials per the paper's Appendix B.
    """

    def __init__(self, seed: int = 7, lam=(0.25, 0.25, 0.15, 0.35)):
        self.rng = random.Random(seed)
        self.lam = lam
        self.trials: list[MetaTrial] = []
        self._bounds = [
            (0.0, 2e-3),  # a_urg per-token slope (e.g. 0.002 -> +6 at 3000 tok)
            (0.05, 4.0),  # b_urg intercept
            (0.0, 0.5),   # b_fair intercept
            (1.2, 4.0),   # alpha gap ratio
        ]
        self._next_theta = (0.0, 1.0, 0.05, 2.0)

    def next_theta(self) -> tuple[float, ...]:
        """UCB acquisition on a squared-exponential GP with direct solves."""
        if not self.trials or all(t.reward is None for t in self.trials):
            theta = self._next_theta
            self._next_theta = self._sample()
            return theta
        xs = [t.theta for t in self.trials if t.reward is not None]
        ys = [t.reward for t in self.trials if t.reward is not None]  # type: ignore
        # normalize rewards to zero mean unit variance for kernel conditioning
        mu = sum(ys) / len(ys)
        sd = (sum((y - mu) ** 2 for y in ys) / len(ys)) ** 0.5 or 1.0
        ys_n = [(y - mu) / sd for y in ys]

        def k(a, b):
            d2 = sum((ai - bi) ** 2 for ai, bi in zip(a, b))
            return math.exp(-d2 / (2 * 0.5 ** 2))

        n = len(xs)
        K = [[k(a, b) for b in xs] for a in xs]
        # K + σ_n² I solve (small n: direct Gaussian elimination)
        sn2 = 0.05
        for i in range(n):
            K[i][i] += sn2
        alpha_ = self._solve(K, ys_n)

        def predict(z):
            ks = [k(z, x) for x in xs]
            mean = sum(ki * ai for ki, ai in zip(ks, alpha_))
            var = k(z, z) - sum(
                ki * vi for ki, vi in zip(ks, self._solve(K, ks))
            )
            return mean, max(var, 0.0) ** 0.5

        # candidate grid + kappa=1.2 exploration
        best, best_ucb = None, -1e18
        for _ in range(64):
            z = self._sample()
            m, s = predict(z)
            ucb = m + 1.2 * s
            if ucb > best_ucb:
                best, best_ucb = z, ucb
        return best or self._sample()

    def _sample(self) -> tuple[float, ...]:
        return tuple(
            self.rng.uniform(lo, hi) for lo, hi in self._bounds
        )

    @staticmethod
    def _solve(A, b):
        n = len(A)
        M = [row[:] + [b[i]] for i, row in enumerate(A)]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(M[r][col]))
            M[col], M[piv] = M[piv], M[col]
            d = M[col][col] or 1e-9
            for r in range(n):
                if r != col and M[r][col] != 0:
                    f = M[r][col] / d
                    for c in range(col, n + 1):
                        M[r][c] -= f * M[col][c]
        return [M[i][n] / (M[i][i] or 1e-9) for i in range(n)]

    def record(self, theta, reward, stats):
        self.trials.append(MetaTrial(theta, reward, stats))

    @staticmethod
    def reward_from_stats(stats: dict, lam=(0.25, 0.25, 0.15, 0.35)) -> float:
        """Eq.(5): R = lam1*C + lam2*L - lam3*S - lam4*U from window stats.

        C = -mean_within_spread  (larger spread -> smaller reward term)
        L = -occupancy_cv        (larger imbalance -> smaller reward term)
        S = queue_count_ratio    (queue proliferation penalized)
        U = mean_wait_s          (user-experience wait penalty)
        """
        C = -stats.get("mean_within_spread", 0.0)
        L = -stats.get("occupancy_cv", 0.0)
        S = stats.get("queue_count_ratio", 0.0)
        U = stats.get("mean_wait_s", 0.0)
        return lam[0] * C + lam[1] * L - lam[2] * S - lam[3] * U


class EWSJFMonitor:
    """Ring-buffer monitor of the strategic loop (Section 3.1)."""

    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self._lock = threading.Lock()
        self._items: list[tuple[str, int, float]] = []  # (req_id, prompt_len, wait_s)
        self._ptr = 0

    def observe(self, req_id: str, prompt_len: int, wait_s: float) -> None:
        with self._lock:
            if len(self._items) < self.capacity:
                self._items.append((req_id, prompt_len, wait_s))
            else:
                self._items[self._ptr] = (req_id, prompt_len, wait_s)
                self._ptr = (self._ptr + 1) % self.capacity

    def snapshot_lengths(self) -> list[int]:
        with self._lock:
            return [p for _, p, _ in self._items]

    def snapshot(self) -> list[tuple[str, int, float]]:
        with self._lock:
            return list(self._items)
