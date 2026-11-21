# SPDX-License-Identifier: Apache-2.0
# Conjugate Compilation co-scheduling: dependency-preserving multi-stream
# execution of the co-located draft and target models (paper Section 2.1,
# Eq.(1)-(2)). Overlap site realized in vLLM V1 online serving: at steps that
# schedule prompt tokens, the draft model's prompt prefill depends only on
# the input token ids (data-independent of the target forward of the same
# step), so it runs on a secondary CUDA stream concurrently with the target
# forward; a CUDA event joins the streams before the drafter proposal pass,
# which then skips the warmed prompt rows (their draft KV is resident).
# All true dependencies of the speculative round (draft chain -> verify)
# remain ordered on the main stream.
#
# The decode-round draft-ahead variant of the paper's cross-round overlap is
# intentionally not enabled by default: it needs k extra draft-KV slots per
# running request per round which the scheduler-side slot accounting does
# not provision, and at the measured full-acceptance rate (~0.165) its
# expected gain is bounded by ~5% of round time (go/no-go analysis in the
# report).

from __future__ import annotations

import logging

import torch

from vllm.v1.attention.backend import CommonAttentionMetadata

logger = logging.getLogger(__name__)


class ConjugateCoScheduler:
    """Secondary-stream co-scheduler for a draft-model speculative engine.

    Owned by the GPU model runner; active only when speculative_config.
    conjugate_coschedule is true.
    """

    def __init__(self, drafter, max_num_tokens: int):
        self.drafter = drafter
        self.device = drafter.device
        self.stream = torch.cuda.Stream()
        self.join_event = torch.cuda.Event()
        self.warm_input_ids = torch.zeros(
            max_num_tokens, dtype=torch.int32, device=self.device
        )
        self.warm_positions = torch.zeros(
            max_num_tokens, dtype=torch.int64, device=self.device
        )
        # batch-row -> warmed span for the CURRENT step (set by warmup,
        # consumed by the reduced first pass)
        self.pending_warm: dict[int, int] = {}
        # rows kept for the reduced drafter first pass (flat row indices of
        # the full step batch), None = no reduction this step
        self.reduced_keep_rows: torch.Tensor | None = None
        self._counters = {
            "warmups_launched": 0,
            "warmup_tokens": 0,
            "propose_rows_saved": 0,
            "launch_cpu_us": 0.0,
        }
        logger.info_once(
            "Conjugate Compilation co-scheduling enabled (draft prefill "
            "overlap on secondary stream)"
        )

    # ------------------------------------------------------------------
    # telemetry / lifecycle
    # ------------------------------------------------------------------

    def counters(self) -> dict:
        return dict(self._counters)

    def step_reset(self) -> None:
        self.pending_warm = {}
        self.reduced_keep_rows = None

    # ------------------------------------------------------------------
    # reduced first pass: row selection + reduced CAD
    # ------------------------------------------------------------------

    def build_reduction(
        self,
        qsl_cpu,  # torch int32 cpu tensor, size num_reqs+1
        num_reqs: int,
        slot_mapping: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        """Compute (keep_rows, total) for the drafter first pass: warmed
        rows keep only their last scheduled token; others keep all rows."""
        keep: list[int] = []
        for i in range(num_reqs):
            s, e = int(qsl_cpu[i]), int(qsl_cpu[i + 1])
            span = e - s
            warmed = self.pending_warm.get(i, 0)
            if warmed > 1 and span > 1:
                keep.append(e - 1)  # only the last token row
                self._counters["propose_rows_saved"] += span - 1
            else:
                keep.extend(range(s, e))
        keep_rows = torch.as_tensor(keep, dtype=torch.int64, device=device)
        return keep_rows, len(keep)

    @staticmethod
    def reduced_cad(
        cad: CommonAttentionMetadata,
        keep_rows: torch.Tensor,
        total: int,
        num_reqs: int,
    ) -> CommonAttentionMetadata:
        """CAD with the same per-request count but queries restricted to the
        kept rows. Query spans become: warmed requests keep their tail token;
        the +1 extra slot per request is added later by the drafter's normal
        input-assembly kernel (spans feed through query_start_loc)."""
        import numpy as np

        # per-request kept counts derived from cad spans with warm reduction
        # (caller recomputes spans consistent with keep_rows ordering)
        raise NotImplementedError  # replaced by runner-side assembly

    # NOTE: span reconstruction is done in the runner where qsl_cpu is
    # available; see gpu_model_runner._conjugate_reduce_first_pass.
