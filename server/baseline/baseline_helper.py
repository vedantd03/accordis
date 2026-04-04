"""Baseline inference helper for the Accordis BFT consensus environment.

Runs a static (no-op) agent or an LLM-based agent against all three task
difficulties (easy, medium, hard) and reports scores.

Supported providers:
    static   — keeps STATIC_BASELINE_CONFIG unchanged every step (no LLM needed)
    openai   — OpenAI chat completion (requires OPENAI_API_KEY or HF_TOKEN + API_KEY)
    gemini   — Google Gemini (requires GEMINI_API_KEY or HF_TOKEN + API_KEY)
"""

from __future__ import annotations

import asyncio
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

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an RL agent tuning Byzantine Fault-Tolerant (BFT) consensus parameters
    for a cluster of honest nodes. Your goal is to maximise transaction throughput
    and liveness while keeping the number of view changes low.

    At each step you receive JSON observations for every honest node and must return
    a JSON object mapping each node_id to its new BFT configuration.

    Parameters and their safe ranges:
      view_timeout_ms:             200 - 10000  (ms before triggering a view change)
      pipeline_depth:              1 - 8        (concurrent in-flight rounds)
      replication_batch_size:      1 - 512      (txns per proposal)
      equivocation_threshold:      1 - 15       (votes before declaring equivocation)
      vote_aggregation_timeout_ms: 50 - 1000    (must be strictly less than view_timeout_ms / 2)

    Respond with ONLY valid JSON — no prose, no code fences, no markdown. Example:
    {
      "node_0": {
        "view_timeout_ms": 2000,
        "pipeline_depth": 2,
        "replication_batch_size": 64,
        "equivocation_threshold": 5,
        "vote_aggregation_timeout_ms": 500
      }
    }
    """
).strip()


def _select_task(name: str):
    name = name.lower()
    if name == "medium":
        return MediumTask()
    if name == "hard":
        return HardTask()
    return EasyTask(curriculum_level=1)


def _obs_to_dict(obs_nodes: Dict[NodeID, AccordisObservation]) -> str:
    summary = {}
    for nid, o in obs_nodes.items():
        summary[nid] = {
            "role":                  o.current_role.value,
            "view":                  o.current_view,
            "commit_tps":            round(o.commit_throughput_tps, 2),
            "pending_txns":          o.pending_txn_count,
            "pipeline_utilisation":  round(o.pipeline_utilisation, 2),
            "qc_miss_streak":        o.qc_formation_miss_streak,
            "view_changes_recent":   o.view_change_count_recent,
            "current_config": {
                "view_timeout_ms":             o.current_config.view_timeout_ms,
                "pipeline_depth":              o.current_config.pipeline_depth,
                "replication_batch_size":      o.current_config.replication_batch_size,
                "equivocation_threshold":      o.current_config.equivocation_threshold,
                "vote_aggregation_timeout_ms": o.current_config.vote_aggregation_timeout_ms,
            },
        }
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

        Return your BFT configuration for every node listed above.
        """
    ).strip()

    raw_configs: Dict = {}
    try:
        text = await asyncio.to_thread(llm.complete, SYSTEM_PROMPT, user_prompt)
        raw_configs = json.loads(text)
    except Exception:
        pass

    node_actions: Dict[NodeID, AccordisAction] = {}
    for nid in node_ids:
        cfg_raw  = raw_configs.get(nid, {})
        prev_cfg = obs.nodes[nid].current_config
        node_actions[nid] = AccordisAction(
            node_id=nid,
            view_timeout_ms=int(cfg_raw.get("view_timeout_ms", prev_cfg.view_timeout_ms)),
            pipeline_depth=int(cfg_raw.get("pipeline_depth", prev_cfg.pipeline_depth)),
            replication_batch_size=int(cfg_raw.get("replication_batch_size", prev_cfg.replication_batch_size)),
            equivocation_threshold=int(cfg_raw.get("equivocation_threshold", prev_cfg.equivocation_threshold)),
            vote_aggregation_timeout_ms=int(cfg_raw.get("vote_aggregation_timeout_ms", prev_cfg.vote_aggregation_timeout_ms)),
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
        import os
        os.environ.setdefault("PROVIDER", provider)
        if model:
            os.environ["MODEL_NAME"] = model
        llm = LLMClientFactory.create()

    total_reward = 0.0
    steps_taken  = 0
    score        = 0.0
    last_reward  = 0.0

    try:
        obs: MultiNodeObservation = env.reset(**conds)
        episode_max_steps = conds.get("max_steps", 200)

        for step in range(1, episode_max_steps + 1):
            if provider == "static":
                action = _get_static_action(obs)
            else:
                action = await _get_llm_action(llm, step, obs, last_reward)

            obs = env.step(action)

            reward       = float(obs.reward) if obs.reward is not None else 0.0
            done         = bool(obs.done)
            last_reward  = reward
            total_reward += reward
            steps_taken   = step

            if done:
                break

        if env._episode_log:
            score = task.grade(env._episode_log)

    finally:
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass

    return {
        "task":         task_name,
        "steps":        steps_taken,
        "total_reward": round(total_reward, 2),
        "score":        round(score, 4),
    }


async def run_baseline(
    tasks: Optional[list[str]] = None,
    provider: str = "static",
    model: Optional[str] = None,
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
