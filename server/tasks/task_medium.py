"""MediumTask — Level 3–5 task definition.

Goal: Recover from crash failures and maintain consistency under adaptive delays.
"""

from __future__ import annotations

from accordis.models import EpisodeLog, LeaderRotation
from accordis.server.tasks.base_task import BaseTask
from accordis.server.oracle.verifier import CorrectnessOracle


def _safe_score(score: float) -> float:
    return round(max(0.0, min(1.0, float(score))), 3)


class MediumTask(BaseTask):
    """Level 3–5 task.

    Initial conditions:
      curriculum_level: 3–5
      n_nodes:          7
      f_byzantine:      2 (1 crash + 1 Byzantine)
      leader_rotation:  round_robin
      network_profile:  FaultProfile per level
      bfa_strategy:     SELECTIVE_DELAY / EQUIVOCATION / ADAPTIVE_MIRROR
      max_steps:        100

    Success criteria:
      liveness_rate >= 0.80, fast_leader_recovery >= 1, no oracle violations

    Grader:
      recovery_bonus = 0.2 if fast_leader_recovery_count >= 1 else 0.0
      score = 0.4 × liveness_rate
            + 0.2 × max(0, 1 - view_change_count / 10)
            + recovery_bonus
            + 0.2 × (1.0 if agreement_ok and validity_ok else 0.0)
    """

    task_id           = "medium"
    curriculum_levels = [3, 4, 5]
    n_nodes           = 7
    f_byzantine       = 2
    leader_rotation   = LeaderRotation.ROUND_ROBIN
    max_steps         = 100

    def __init__(self, curriculum_level: int = 3) -> None:
        self.curriculum_levels = [curriculum_level]

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

        recovery_bonus = 0.2 if fast_leader_recovery_count >= 1 else 0.0
        score = 0.4 × liveness_rate
              + 0.2 × max(0, 1 - view_change_count / 10)
              + recovery_bonus
              + 0.2 × (1.0 if agreement_ok and validity_ok else 0.0)
        """
        if episode_log is None or not getattr(episode_log, "steps", None):
            return 0.0

        try:
            oracle = CorrectnessOracle()
            final_state = episode_log.steps[-1]

            liveness = oracle.check_liveness(final_state)
            verifier = oracle.run_all(final_state)
            correctness = 1.0 if (not verifier.agreement_violated and not verifier.validity_violated) else 0.0

            view_change_count = getattr(final_state, "view_change_count", 0)
            vc_penalty = max(0.0, 1.0 - view_change_count / 10.0)

            fast_recovery_count = sum(
                1
                for reward in getattr(episode_log, "rewards", [])
                if getattr(reward, "fast_leader_recovery", 0) > 0
            )
            recovery_bonus = 0.2 if fast_recovery_count >= 1 else 0.0

            score = (
                0.4 * getattr(liveness, "liveness_rate", 0.0)
                + 0.2 * vc_penalty
                + recovery_bonus
                + 0.2 * correctness
            )
            return _safe_score(score)
        except Exception:
            return 0.0
