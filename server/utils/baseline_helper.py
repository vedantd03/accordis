"""Baseline inference helper for the Accordis BFT consensus environment.

Runs a static (no-op) agent or an LLM-based agent against all three task
difficulties (easy, medium, hard) and reports scores.

Supported providers:
    static   — keeps STATIC_BASELINE_CONFIG unchanged every step (no LLM needed)
    openai   — OpenAI chat completion (requires OPENAI_API_KEY or HF_TOKEN + API_KEY)
    gemini   — Google Gemini (requires GEMINI_API_KEY or HF_TOKEN + API_KEY)
"""

from __future__ import annotations

import os
import json
import textwrap
from typing import Dict, Optional

from accordis.llm_factory import BaseLLMClient, LLMClientFactory
from accordis.models import (
    AccordisAction,
    AccordisObservation,
    MultiNodeAction,
    MultiNodeObservation,
    NodeID,
    STATIC_BASELINE_CONFIG,
)
from accordis.server.accordis_environment import AccordisEnvironment
from accordis.server.adapters import create_adapter
from accordis.server.tasks.task_easy import EasyTask
from accordis.server.tasks.task_medium import MediumTask
from accordis.server.tasks.task_hard import HardTask

from accordis.server.utils.logger import logger
from accordis.server.utils.constants import (
    SYSTEM_PROMPT
)


def _select_task(name: str):
    name = name.lower()
    if name == "medium":
        return MediumTask()
    if name == "hard":
        return HardTask()
    return EasyTask(curriculum_level=1)


def _obs_to_dict(obs_nodes: Dict[NodeID, AccordisObservation]) -> str:
    """Compact JSON summary of all honest-node observations for the LLM.

    Includes a top-level cluster_min_pending field: the minimum pending_txns
    across all honest nodes. This is derived entirely from per-node observable
    data and gives the best lower bound on remaining cluster work.
    """
    node_summary = {}
    for nid, o in obs_nodes.items():
        node_summary[nid] = {
            "role":                   o.current_role.value,
            "view":                   o.current_view,
            "commit_tps":             round(o.commit_throughput_tps, 2),
            "pending_txns":           o.pending_txn_count,
            "pipeline_utilisation":   round(o.pipeline_utilisation, 2),
            "qc_miss_streak":         o.qc_formation_miss_streak,
            "view_changes_recent":    o.view_change_count_recent,
            "suspected_byzantine":    o.suspected_byzantine,
            "current_config": {
                "view_timeout_ms":             o.current_config.view_timeout_ms,
                "pipeline_depth":              o.current_config.pipeline_depth,
                "replication_batch_size":      o.current_config.replication_batch_size,
                "equivocation_threshold":      o.current_config.equivocation_threshold,
                "vote_aggregation_timeout_ms": o.current_config.vote_aggregation_timeout_ms,
            },
        }

    cluster_min_pending = min(o.pending_txn_count for o in obs_nodes.values())

    summary = {"cluster_min_pending": cluster_min_pending, "nodes": node_summary}
    return json.dumps(summary, indent=2)


def _get_static_action(obs: MultiNodeObservation) -> MultiNodeAction:
    """Return STATIC_BASELINE_CONFIG for every honest node — no changes."""
    node_actions: Dict[NodeID, AccordisAction] = {}
    for nid in obs.nodes:
        node_actions[nid] = AccordisAction(
            node_id=nid,
            view_timeout_ms=STATIC_BASELINE_CONFIG.view_timeout_ms,
            pipeline_depth=STATIC_BASELINE_CONFIG.pipeline_depth,
            replication_batch_size=STATIC_BASELINE_CONFIG.replication_batch_size,
            equivocation_threshold=STATIC_BASELINE_CONFIG.equivocation_threshold,
            vote_aggregation_timeout_ms=STATIC_BASELINE_CONFIG.vote_aggregation_timeout_ms,
        )
    return MultiNodeAction(nodes=node_actions)


async def _get_llm_action(
    llm: BaseLLMClient,
    step: int,
    obs: MultiNodeObservation,
    last_reward: float,
) -> MultiNodeAction:
    node_ids = list(obs.nodes.keys())

    user_prompt = textwrap.dedent(
        f"""
        Step: {step}
        Last reward: {last_reward:.2f}
        Current observations:
        {_obs_to_dict(obs.nodes)}

        Return your BFT configuration for every node_id found under "nodes" above.
        """
    ).strip()

    raw_configs: Dict = {}
    try:
        # text = await asyncio.to_thread(llm.complete, SYSTEM_PROMPT, user_prompt)
        text = await llm.complete(SYSTEM_PROMPT, user_prompt)
        raw_configs = json.loads(text)
    except Exception as exc:
        logger.debug(f"[DEBUG] LLM request or JSON parse failed: {exc}", exc_info=True)

    node_actions: Dict[NodeID, AccordisAction] = {}
    for nid in node_ids:
        cfg_raw   = raw_configs.get(nid, {})
        prev_cfg  = obs.nodes[nid].current_config
        node_actions[nid] = AccordisAction(
            node_id=nid,
            view_timeout_ms=int(
                cfg_raw.get("view_timeout_ms", prev_cfg.view_timeout_ms)
            ),
            pipeline_depth=int(
                cfg_raw.get("pipeline_depth", prev_cfg.pipeline_depth)
            ),
            replication_batch_size=int(
                cfg_raw.get("replication_batch_size", prev_cfg.replication_batch_size)
            ),
            equivocation_threshold=int(
                cfg_raw.get("equivocation_threshold", prev_cfg.equivocation_threshold)
            ),
            vote_aggregation_timeout_ms=int(
                cfg_raw.get("vote_aggregation_timeout_ms", prev_cfg.vote_aggregation_timeout_ms)
            ),
        )

    return MultiNodeAction(nodes=node_actions)


async def _run_single_task(
    task_name: str,
    provider: str,
    model: Optional[str],
) -> dict:
    """Run one full episode for the given task and return a result dict."""
    task  = _select_task(task_name)
    conds = task.get_initial_conditions()

    env = AccordisEnvironment(adapter=create_adapter())

    llm: Optional[BaseLLMClient] = None
    if provider != "static":
        llm = LLMClientFactory.create(provider=provider, model=model)

    log_start(task_name, os.getenv("ACCORDIS_ADAPTER", "simulated"))

    total_reward = 0.0
    steps_taken  = 0
    score        = 0.0

    try:
        obs: MultiNodeObservation = env.reset(**conds)
        log_observation(obs)
        last_reward  = 0.0

        episode_max_steps = conds.get("max_steps", int(os.getenv("ACCORDIS_MAX_STEPS", "100")))
        for step in range(1, episode_max_steps + 1):
            if provider == "static":
                action = _get_static_action(obs)
            else:
                action = await _get_llm_action(llm, step, obs, last_reward)
            log_action(action)
            obs = env.step(action)
            log_observation(obs)

            reward       = float(obs.reward) if obs.reward is not None else 0.0
            done         = bool(obs.done)
            last_reward  = reward
            total_reward += reward
            steps_taken   = step

            log_step(step=step, reward=reward, total=total_reward, done=done)

            if done:
                break

        if env._episode_log:
            score = task.grade(env._episode_log)

    except Exception as exc:
        logger.debug(f"[DEBUG] Episode error: {exc}", exc_info=True)

    finally:
        if llm is not None:
            try:
                await llm.close()
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass
        log_end(steps=steps_taken, total_reward=total_reward, score=score)

    return {
        "task":         task_name,
        "steps":        steps_taken,
        "total_reward": round(total_reward, 2),
        "score":        round(score, 4),
    }


async def run_baseline(
    provider: str,
    model: Optional[str],
    tasks: Optional[list[str]] = None,
) -> dict:
    """Run baseline evaluation across one or more task difficulties.

    Args:
        tasks:    List of task names to run. Defaults to all three if None.
        provider: Inference provider — "static", "openai", or "gemini".
        model:    LLM model name (required when provider is not "static").

    Returns:
        A dict with status and per-task results.
    """
    if tasks is None:
        tasks = ["easy", "medium", "hard"]

    results = {}
    for task_name in tasks:
        results[task_name] = await _run_single_task(task_name, provider, model)

    return {
        "status":   "success",
        "provider": provider,
        "tasks":    tasks,
        "data":     results,
    }

# ── Logging ────────────────────────────────────────────────────────────────────

def log_start(task: str, adapter: str) -> None:
    logger.info(f"[START] task={task} adapter={adapter}")


def log_step(step: int, reward: float, total: float, done: bool) -> None:
    logger.info(
        f"[STEP]  step={step} reward={reward:.1f} total={total:.1f} done={done}"
    )

def log_action(action: MultiNodeAction) -> None:
    action_dict = {nid: {
        "view_timeout_ms": a.view_timeout_ms,
        "pipeline_depth": a.pipeline_depth,
        "replication_batch_size": a.replication_batch_size,
        "equivocation_threshold": a.equivocation_threshold,
        "vote_aggregation_timeout_ms": a.vote_aggregation_timeout_ms,
    } for nid, a in action.nodes.items()}
    logger.info(f"[ACTION] {json.dumps(action_dict)}")

def log_observation(obs: MultiNodeObservation) -> None:
    logger.info(f"[OBSERVATION] {_obs_to_dict(obs.nodes)}")

def log_end(steps: int, total_reward: float, score: float) -> None:
    logger.info(
        f"[END]   steps={steps} total_reward={total_reward:.1f} score={score:.2f}"
    )