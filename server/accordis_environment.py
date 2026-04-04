"""AccordisEnvironment — main RL environment implementation.

This class is the single orchestrator between the OpenEnv API layer and the
consensus engine. It accepts a BaseConsensusAdapter at construction and has
ZERO imports from server/adapters/simulated/ or server/adapters/librabft/.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

from accordis.models import (
    AccordisAction,
    AccordisObservation,
    AccordisReward,
    AccordisRubric,
    AccordisState,
    AccordisTransform,
    BFAStrategy,
    BFTConfig,
    Block,
    EpisodeLog,
    LeaderRotation,
    LivenessResult,
    MultiNodeAction,
    MultiNodeObservation,
    NodeID,
    NodeRole,
    NodeState,
    ProposalRegistry,
    SAFE_BFT_TUNING_BOUNDS,
    STATIC_BASELINE_CONFIG,
    Transaction,
)
from accordis.server.adapters.base import BaseConsensusAdapter
from accordis.server.adversary.bfa import ByzantineFailureAgent
from accordis.server.oracle.verifier import CorrectnessOracle
from accordis.server.rewards.reward_calculator import RewardCalculator
from accordis.server.curriculum.manager import CurriculumManager
from accordis.server.adapters import create_adapter


class AccordisEnvironment(Environment):
    """Self-adaptive BFT consensus tuning RL environment.

    Accepts a BaseConsensusAdapter at construction. Never constructs or imports
    adapter implementation classes directly.
    """

    DEFAULT_MAX_STEPS   = 200
    DEFAULT_N_NODES     = 4
    DEFAULT_F_BYZANTINE = 0
    DEFAULT_CURRICULUM  = 1
    DEFAULT_BFA_SEED    = 42

    def __init__(self, adapter: Optional[BaseConsensusAdapter] = None) -> None:
        self._adapter    = adapter if adapter is not None else create_adapter()
        self._bfa        = ByzantineFailureAgent()
        self._oracle     = CorrectnessOracle()
        self._calculator = RewardCalculator()
        self._curriculum = CurriculumManager()
        self._transform  = AccordisTransform()

        self._state: AccordisState = AccordisState(episode_id=str(uuid.uuid4()), step=0)
        self._episode_rewards: List[AccordisReward] = []
        self._episode_log: Optional[EpisodeLog] = None
        self._max_steps: int = self.DEFAULT_MAX_STEPS
        self._bfa_seed: int = self.DEFAULT_BFA_SEED

        super().__init__(
            transform=self._transform,
            rubric=AccordisRubric(),
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        *,
        n_nodes: int = DEFAULT_N_NODES,
        f_byzantine: int = DEFAULT_F_BYZANTINE,
        leader_rotation: LeaderRotation = LeaderRotation.ROUND_ROBIN,
        curriculum_level: Optional[int] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        bfa_strategy_seed: int = DEFAULT_BFA_SEED,
    ) -> MultiNodeObservation:
        """Start a new episode and return per-node initial observations.

        reset() Internal Sequence:
          1. self._adapter.stop_cluster()
          2. self._adapter.configure_network(curriculum_level, [])
          3. node_ids = self._adapter.start_cluster(n_nodes, f_byzantine, leader_rotation)
          4. for each honest node: self._adapter.apply_config(node_id, STATIC_BASELINE_CONFIG)
          5. bfa_strategy = self._bfa.select_strategy(curriculum_level, seed=bfa_strategy_seed)
          6. for each honest node: obs[node_id] = self._transform.transform(raw, ...)
          7. Initialise AccordisState, episode_txn_pool, episode_rewards = []
          8. Return obs dict
        """
        self._reset_rubric()

        self._max_steps = max_steps
        self._bfa_seed  = seed if seed is not None else bfa_strategy_seed

        level = curriculum_level if curriculum_level is not None else self._curriculum.level

        # Step 1
        self._adapter.stop_cluster()

        # Step 2
        self._adapter.configure_network(level, [])

        # Step 3
        self._adapter.start_cluster(n_nodes, f_byzantine, leader_rotation)
        honest_nodes    = self._adapter.get_honest_nodes()
        byzantine_nodes = self._adapter.get_byzantine_nodes()

        # Step 4
        for nid in honest_nodes:
            self._adapter.apply_config(nid, STATIC_BASELINE_CONFIG)

        # Step 5
        bfa_strategy = self._bfa.select_strategy(
            curriculum_level=level,
            step=0,
            seed=bfa_strategy_seed,
        )

        # Step 6
        obs: Dict[NodeID, AccordisObservation] = {}
        for nid in honest_nodes:
            raw = self._adapter.read_observation(nid)
            obs[nid] = self._transform.transform(
                raw, nid, step=0, current_config=STATIC_BASELINE_CONFIG
            )

        # Step 7
        eid = episode_id or str(uuid.uuid4())

        node_states: Dict[NodeID, NodeState] = {}
        for nid in honest_nodes:
            node_states[nid] = NodeState(
                node_id=nid,
                is_byzantine=False,
                committed_log=[],
                current_view=0,
                current_role=NodeRole.REPLICA,
                config=STATIC_BASELINE_CONFIG,
            )
        for nid in byzantine_nodes:
            node_states[nid] = NodeState(
                node_id=nid,
                is_byzantine=True,
                committed_log=[],
                current_view=0,
                current_role=NodeRole.REPLICA,
                config=STATIC_BASELINE_CONFIG,
            )

        txn_pool = [Transaction(id=f"tx_{i}", submitted_at=0) for i in range(1000)]
        registry = ProposalRegistry(
            honest_proposals={tx.id: tx for tx in txn_pool}
        )

        self._state = AccordisState(
            episode_id=eid,
            step=0,
            curriculum_level=level,
            n_nodes=n_nodes,
            f_byzantine=f_byzantine,
            leader_rotation=leader_rotation,
            node_states=node_states,
            view_change_count=0,
            bfa_strategy=bfa_strategy,
            proposal_registry=registry,
            episode_txn_pool=txn_pool,
        )

        self._episode_rewards = []
        self._episode_log = EpisodeLog(
            episode_id=eid,
            curriculum_level=level,
            bfa_strategy=bfa_strategy,
            bfa_strategy_seed=bfa_strategy_seed,
        )

        # Step 8
        return self._apply_transform(obs)

    def step(
        self,
        action: MultiNodeAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MultiNodeObservation:
        """Advance the environment by one synchronous consensus round."""
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")

        prev_state = deepcopy(self._state)
        step = self._state.step + 1
        honest_nodes    = self._adapter.get_honest_nodes()
        byzantine_nodes = self._adapter.get_byzantine_nodes()

        # Step 1+2: Clamp and apply configs
        clamped_configs: Dict[NodeID, BFTConfig] = {}
        for nid, node_action in action.nodes.items():
            if nid in honest_nodes:
                clamped = self._clamp_action(node_action)
                config = BFTConfig(
                    view_timeout_ms=clamped.view_timeout_ms,
                    pipeline_depth=clamped.pipeline_depth,
                    replication_batch_size=clamped.replication_batch_size,
                    equivocation_threshold=clamped.equivocation_threshold,
                    vote_aggregation_timeout_ms=clamped.vote_aggregation_timeout_ms,
                )
                clamped_configs[nid] = config
                self._adapter.apply_config(nid, config)
                if nid in self._state.node_states:
                    self._state.node_states[nid].config = config

        # Step 3: BFA strategy selection
        strategy = self._bfa.select_strategy(
            curriculum_level=self._state.curriculum_level,
            step=step,
            agent_configs=clamped_configs,
            seed=self._bfa_seed,
        )
        params = self._bfa.get_disruption_parameters(
            strategy=strategy,
            agent_configs=clamped_configs,
            target_nodes=honest_nodes,
        )

        # Step 4: Inject Byzantine actions
        for byz_nid in byzantine_nodes:
            self._adapter.inject_byzantine_action(
                byzantine_node_id=byz_nid,
                strategy=strategy,
                target_nodes=honest_nodes,
                parameters=params,
            )

        # Step 5: Advance one consensus step
        self._adapter.advance_one_step()

        # Step 6: Read observations
        obs: Dict[NodeID, AccordisObservation] = {}
        for nid in honest_nodes:
            raw = self._adapter.read_observation(nid)
            config = clamped_configs.get(nid, STATIC_BASELINE_CONFIG)
            obs[nid] = self._transform.transform(raw, nid, step=step, current_config=config)

        # Step 7: Update AccordisState
        view_change_total = 0
        for nid in honest_nodes:
            committed_log_dicts = self._adapter.get_committed_log(nid)
            blocks = [Block(**b) for b in committed_log_dicts]
            if nid in self._state.node_states:
                self._state.node_states[nid].committed_log = blocks
                obs_n = obs.get(nid)
                if obs_n:
                    self._state.node_states[nid].current_view = obs_n.current_view
                    self._state.node_states[nid].current_role = obs_n.current_role
                    view_change_total = max(view_change_total, obs_n.view_change_count_recent)

        self._state.step = step
        self._state.view_change_count = view_change_total
        self._state.bfa_strategy = strategy
        self._state.finalized_txn_count = self._adapter.get_finalized_txn_count()

        # Step 8: Correctness oracle
        verifier_results = self._oracle.run_all(self._state)

        # Step 9: Liveness
        liveness = self._oracle.check_liveness(self._state)

        # Step 11: Done check
        done = (step >= self._max_steps) or (liveness.pending_count == 0)

        # Step 10+12: Reward computation
        if done:
            self._curriculum.record_episode(liveness.liveness_rate)
            if self._curriculum.should_advance():
                self._curriculum.advance()

            if self._episode_log:
                self._episode_log.steps.append(deepcopy(self._state))
                baseline = self._oracle.compute_baseline_comparison(
                    episode_log=self._episode_log,
                    static_config=STATIC_BASELINE_CONFIG,
                    bfa_strategy_seed=self._bfa_seed,
                )
            else:
                baseline = None

            reward = self._calculator.compute(
                prev_state=prev_state,
                current_state=self._state,
                verifier_results=verifier_results,
                baseline=baseline,
                liveness=liveness,
                is_done=True,
            )
        else:
            reward = self._calculator.compute(
                prev_state=prev_state,
                current_state=self._state,
                verifier_results=verifier_results,
                baseline=None,
                liveness=liveness,
                is_done=False,
            )

        self._episode_rewards.append(reward)
        if self._episode_log and not done:
            self._episode_log.steps.append(deepcopy(self._state))
            self._episode_log.rewards.append(reward)

        info = {
            "step":              step,
            "liveness_rate":     liveness.liveness_rate,
            "view_change_count": self._state.view_change_count,
            "bfa_strategy":      strategy.value,
            "agreement_ok":      verifier_results.agreement.passed,
            "validity_ok":       verifier_results.validity.passed,
            "reward_breakdown":  reward.model_dump(),
        }

        for nid in honest_nodes:
            if nid in obs:
                obs[nid].reward = reward.total
                obs[nid].done = done

        rubric_reward = self._apply_rubric(action, obs)
        result = self._apply_transform(obs)
        result.reward = rubric_reward
        result.done = done
        result.metadata = info
        return result

    @property
    def state(self) -> AccordisState:
        if self._state is None:
            raise RuntimeError("reset() must be called before step()")
        return self._state

    # ── Internal Helpers ──────────────────────────────────────────────────────

    def _clamp_action(self, action: AccordisAction) -> AccordisAction:
        """Clamp all tunable fields to SAFE_BFT_TUNING_BOUNDS.

        Also enforces: vote_aggregation_timeout_ms < view_timeout_ms // 2
        The adapter always receives in-bounds values.
        """
        def clamp(val: int, lo: int, hi: int) -> int:
            return max(lo, min(hi, val))

        vt_lo, vt_hi = SAFE_BFT_TUNING_BOUNDS["view_timeout_ms"]
        pd_lo, pd_hi = SAFE_BFT_TUNING_BOUNDS["pipeline_depth"]
        rb_lo, rb_hi = SAFE_BFT_TUNING_BOUNDS["replication_batch_size"]
        et_lo, et_hi = SAFE_BFT_TUNING_BOUNDS["equivocation_threshold"]
        va_lo, va_hi = SAFE_BFT_TUNING_BOUNDS["vote_aggregation_timeout_ms"]

        view_timeout_ms             = clamp(action.view_timeout_ms,             vt_lo, vt_hi)
        pipeline_depth              = clamp(action.pipeline_depth,              pd_lo, pd_hi)
        replication_batch_size      = clamp(action.replication_batch_size,      rb_lo, rb_hi)
        equivocation_threshold      = clamp(action.equivocation_threshold,      et_lo, et_hi)
        vote_aggregation_timeout_ms = clamp(action.vote_aggregation_timeout_ms, va_lo, va_hi)

        max_vat = view_timeout_ms // 2 - 1
        vote_aggregation_timeout_ms = min(vote_aggregation_timeout_ms, max(va_lo, max_vat))

        return AccordisAction(
            node_id=action.node_id,
            view_timeout_ms=view_timeout_ms,
            pipeline_depth=pipeline_depth,
            replication_batch_size=replication_batch_size,
            equivocation_threshold=equivocation_threshold,
            vote_aggregation_timeout_ms=vote_aggregation_timeout_ms,
            suspect_node=action.suspect_node,
            clear_suspicion=action.clear_suspicion,
        )

    def _build_observation(self) -> AccordisObservation:
        """Project the hidden full-system state into the first honest node's view."""
        if self._state is None:
            raise RuntimeError("reset() must be called first")
        honest = self._adapter.get_honest_nodes()
        if not honest:
            raise RuntimeError("No honest nodes available")
        nid = honest[0]
        raw = self._adapter.read_observation(nid)
        config = (
            self._state.node_states[nid].config
            if nid in self._state.node_states
            else STATIC_BASELINE_CONFIG
        )
        return self._transform.transform(raw, nid, step=self._state.step, current_config=config)
