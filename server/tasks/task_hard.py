"""HardTask — Level 6–8 task definition.

Goal: Maintain correctness and liveness under coordinated full-coalition Byzantine attacks.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models import EpisodeLog, LeaderRotation
from server.tasks.base_task import BaseTask
from server.oracle.verifier import CorrectnessOracle


class HardTask(BaseTask):
    """Level 6–8 task.

    Initial conditions:
      curriculum_level: 6–8
      n_nodes:          10
      f_byzantine:      3 (maximum BFT tolerance at n=10)
      leader_rotation:  round_robin (level 6) → vrf / reputation_weighted (levels 7–8)
      network_profile:  FaultProfile per level
      bfa_strategy:     LEADER_SUPPRESS + CASCADE_TIMING + ADAPTIVE_MIRROR (coalition)
      max_steps:        500

    Success criteria:
      liveness_rate >= 0.70, throughput_improvement > 0, no oracle violations

    Grader:
      baseline_delta   = relative_tps_improvement from BaselineComparison
      throughput_score = min(1.0, max(0.0, baseline_delta + 0.5))
      score = 0.35 × liveness_rate
            + 0.25 × throughput_score
            + 0.25 × max(0, 1 - view_change_count / 20)
            + 0.15 × (1.0 if agreement_ok and validity_ok else 0.0)
    """

    task_id           = "hard"
    curriculum_levels = [6, 7, 8]
    n_nodes           = 10
    f_byzantine       = 3
    leader_rotation   = LeaderRotation.ROUND_ROBIN
    max_steps         = 500

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
        }

    def grade(self, episode_log: EpisodeLog) -> float:
        """Score the episode.

        baseline_delta   = relative_tps_improvement from BaselineComparison
        throughput_score = min(1.0, max(0.0, baseline_delta + 0.5))
        score = 0.35 × liveness_rate
              + 0.25 × throughput_score
              + 0.25 × max(0, 1 - view_change_count / 20)
              + 0.15 × (1.0 if agreement_ok and validity_ok else 0.0)
        """
        if not episode_log.steps:
            return 0.0

        oracle = CorrectnessOracle()
        final_state = episode_log.steps[-1]

        liveness  = oracle.check_liveness(final_state)
        verifier  = oracle.run_all(final_state)
        correctness = 1.0 if (not verifier.agreement_violated and not verifier.validity_violated) else 0.0

        view_change_count = final_state.view_change_count
        vc_penalty = max(0.0, 1.0 - view_change_count / 20.0)

        # Compute baseline comparison from episode log
        baseline = oracle.compute_baseline_comparison(
            episode_log=episode_log,
            static_config=None,  # type: ignore[arg-type]
            bfa_strategy_seed=episode_log.bfa_strategy_seed,
        )
        baseline_delta   = baseline.relative_tps_improvement
        throughput_score = min(1.0, max(0.0, baseline_delta + 0.5))

        score = (
            0.35 * liveness.liveness_rate
            + 0.25 * throughput_score
            + 0.25 * vc_penalty
            + 0.15 * correctness
        )
        return max(0.0, min(1.0, score))
