"""EasyTask — Level 1–2 task definition.

Goal: Achieve and maintain consensus under stable conditions.
"""

from __future__ import annotations

from accordis.models import EpisodeLog, LeaderRotation
from accordis.server.tasks.base_task import BaseTask
from accordis.server.oracle.verifier import CorrectnessOracle


class EasyTask(BaseTask):
    """Level 1–2 task.

    Initial conditions:
      curriculum_level: 1 or 2
      n_nodes:          4
      f_byzantine:      0 (level 1) / 1 (level 2, RANDOM_DELAY only)
      leader_rotation:  round_robin
      network_profile:  FaultProfile(p50=2ms, p99=5ms, jitter=1ms, loss=0%)
      bfa_strategy:     NONE / RANDOM_DELAY
      max_steps:        200

    Success criteria:
      liveness_rate >= 0.95, view_change_count <= 3, no oracle violations

    Grader:
      score = 0.5 × liveness_rate
            + 0.3 × max(0, 1 - view_change_count / 5)
            + 0.2 × (1.0 if agreement_ok and validity_ok else 0.0)
      Clamp to [0.0, 1.0]
    """

    task_id           = "easy"
    curriculum_levels = [1, 2]
    n_nodes           = 4
    f_byzantine       = 0  # overridden per level
    leader_rotation   = LeaderRotation.ROUND_ROBIN
    max_steps         = 200

    def __init__(self, curriculum_level: int = 1) -> None:
        self.curriculum_levels = [curriculum_level]
        self.f_byzantine = 0 if curriculum_level == 1 else 1

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

        score = 0.5 × liveness_rate
              + 0.3 × max(0, 1 - view_change_count / 5)
              + 0.2 × (1.0 if agreement_ok and validity_ok else 0.0)
        """
        if not episode_log.steps:
            return 0.0

        oracle = CorrectnessOracle()
        final_state = episode_log.steps[-1]

        liveness    = oracle.check_liveness(final_state)
        verifier    = oracle.run_all(final_state)
        correctness = 1.0 if (not verifier.agreement_violated and not verifier.validity_violated) else 0.0

        view_change_count = final_state.view_change_count
        vc_penalty = max(0.0, 1.0 - view_change_count / 5.0)

        score = (
            0.5 * liveness.liveness_rate
            + 0.3 * vc_penalty
            + 0.2 * correctness
        )
        return max(0.0, min(1.0, score))
