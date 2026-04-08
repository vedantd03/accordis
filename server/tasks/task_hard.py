"""HardTask — Level 6–8 task definition.

Goal: Maintain throughput efficiency under coordinated Byzantine attacks.
      The pool drains within budget even with static config; the score reflects
      how efficiently and stably the agent achieved that.
"""

from __future__ import annotations

from accordis.models import EpisodeLog, LeaderRotation
from accordis.server.tasks.base_task import BaseTask
from accordis.server.oracle.verifier import CorrectnessOracle

_POOL_SIZE = 1_800   # drains in ~85 steps at default batch=64, ~10 steps at batch=512
_MAX_STEPS = 100     # agent budget for LLM cost reasons

# throughput denominator: max possible batch per step (no Byzantine adjustment).
# Byzantine disruption (~30% at level 6) naturally caps achievable throughput at ~0.65,
# so this denominator punishes sub-optimal batch_size without a custom constant.
_THROUGHPUT_DENOM = 170.0

# vc_penalty denominator: generous enough to give partial credit for tuning
# vote_aggregation_timeout_ms to handle SELECTIVE_DELAY (200–800ms delays at level 6).
# Expected Byzantine VCs ≈ 30 in a full 100-step episode; some agents trigger fewer
# false VCs by raising VAT, so denominator=50 creates meaningful discrimination.
_VC_DENOM = 50


class HardTask(BaseTask):
    """Level 6–8 task.

    Initial conditions:
      curriculum_level: 6–8
      n_nodes:          10
      f_byzantine:      2 (coordinated attack on leaders, causing 30% disruption at level 6)
      leader_rotation:  round_robin (level 6) → vrf / reputation_weighted (levels 7–8)
      network_profile:  FaultProfile per level (level 6: p99=300ms, packet_loss=5%)
      bfa_strategy:     LEADER_SUPPRESS + SELECTIVE_DELAY (level 6);
                        CASCADE_TIMING + EQUIVOCATION (level 7);
                        full coalition (level 8)
      pool_size:        2_500 (drains within budget even at default batch=64)
      max_steps:        100

    The two agent-sensitive levers at level 6:
      1. replication_batch_size: default=64, max=512 — 8× throughput difference.
      2. vote_aggregation_timeout_ms: SELECTIVE_DELAY injects up to 800ms delays;
         default VAT=500ms causes QC misses; raising to 900ms+ avoids them.

    Grader:
      liveness_rate    = finalized_txns / pool_size           (always 1.0 if pool drains)
      throughput_score = min(1.0, txns_per_step / _THROUGHPUT_DENOM) (rewards large batch_size)
      vc_penalty       = max(0.0, 1 - view_change_count / 50) (rewards VAT tuning)
      score = 0.05 × liveness_rate
            + 0.75 × throughput_score
            + 0.10 × vc_penalty
            + 0.10 × correctness

    Expected scores (level 6):
      Static defaults (batch=64,  VAT=500ms): ~0.26   (drains at step ~85)
      Median LLM    (batch=200,  VAT=500ms): ~0.42   (drains at step ~27)
      Expert agent  (batch=512,  VAT=1000ms): ~0.76   (drains at step ~10)
    Score crosses 0.5 only when batch ≥ ~310 (60% of max).
    """

    task_id           = "hard"
    curriculum_levels = [6, 7, 8]
    n_nodes           = 10
    f_byzantine       = 2
    leader_rotation   = LeaderRotation.ROUND_ROBIN
    max_steps         = _MAX_STEPS

    def __init__(self, curriculum_level: int = 6) -> None:
        self.curriculum_levels = [curriculum_level]
        # Level 6 → round_robin; levels 7–8 → vrf / reputation_weighted
        if curriculum_level == 7:
            self.leader_rotation = LeaderRotation.VRF
        elif curriculum_level == 8:
            self.leader_rotation = LeaderRotation.REPUTATION_WEIGHTED
        else:
            self.leader_rotation = LeaderRotation.ROUND_ROBIN

    def get_initial_conditions(self) -> dict:
        return {
            "n_nodes":          self.n_nodes,
            "f_byzantine":      self.f_byzantine,
            "leader_rotation":  self.leader_rotation,
            "curriculum_level": self.curriculum_levels[0],
            "max_steps":        self.max_steps,
            "pool_size":        _POOL_SIZE,
        }

    def grade(self, episode_log: EpisodeLog) -> float:
        """Score the episode.

        liveness_rate    = finalized_txns / pool_size           (always 1.0 if pool drains)
        throughput_score = min(1.0, txns_per_step / _THROUGHPUT_DENOM)
        vc_penalty       = max(0.0, 1 - view_change_count / 50)
        score = 0.05 × liveness_rate
              + 0.75 × throughput_score
              + 0.10 × vc_penalty
              + 0.10 × correctness
        """
        if not episode_log.steps:
            return 0.0

        oracle = CorrectnessOracle()
        final_state = episode_log.steps[-1]

        liveness    = oracle.check_liveness(final_state)
        verifier    = oracle.run_all(final_state)
        correctness = 1.0 if (not verifier.agreement_violated and not verifier.validity_violated) else 0.0

        steps = max(1, final_state.step)
        txns_per_step    = liveness.committed_count / steps
        throughput_score = min(1.0, txns_per_step / _THROUGHPUT_DENOM)

        view_change_count = final_state.view_change_count
        vc_penalty = max(0.0, 1.0 - view_change_count / _VC_DENOM)

        score = (
            0.05 * liveness.liveness_rate
            + 0.75 * throughput_score
            + 0.10 * vc_penalty
            + 0.10 * correctness
        )
        return max(0.0, min(1.0, score))
