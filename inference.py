"""
Accordis Inference Script
===================================
MANDATORY environment variables:
    ACCORDIS_ADAPTER   Adapter to use: "simulated" (default) or "librabft".

LLM selection:
    API_KEY / HF_TOKEN     → OpenAI compatible model

Optional:
    ACCORDIS_TASKS      Task difficulty: "easy" (default), "medium", "hard".
    ACCORDIS_MAX_STEPS Override maximum steps per episode (default from task).

STDOUT FORMAT
- The script emits exactly three line types to stdout, in this order:

    [START] task=<task_name> adapter=<adapter>
    [STEP]  step=<n> reward=<r> total=<cumulative> done=<True|False>
    [END]   steps=<n> total_reward=<r> score=<s>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after episode ends, always emitted (even on exception).
    - reward and total_reward are formatted to 1 decimal place.
    - score is formatted to 2 decimal places.
    - done is a Python bool string (True or False).
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=easy adapter=simulated
    [STEP]  step=1 reward=-1.0 total=-1.0 done=False
    [STEP]  step=2 reward=59.0 total=58.0 done=False
    ...
    [END]   steps=87 total_reward=340.0 score=0.73
"""
from __future__ import annotations

import json
import os
import asyncio
import argparse
import textwrap
from dotenv import load_dotenv
load_dotenv()
from typing import Dict, Optional
from abc import ABC, abstractmethod

from accordis.models import (
    AccordisAction,
    AccordisObservation,
    MultiNodeAction,
    MultiNodeObservation,
    NodeID,
    STATIC_BASELINE_CONFIG,
)
# from accordis.server.accordis_environment import AccordisEnvironment
from accordis.client import AccordisEnvironment
from accordis.server.adapters import create_adapter
from accordis.server.tasks.task_easy import EasyTask
from accordis.server.tasks.task_medium import MediumTask
from accordis.server.tasks.task_hard import HardTask

from accordis.server.utils.logger import logger
from accordis.server.utils.constants import (
    SYSTEM_PROMPT
)

IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 

class HuggingFaceClient():
    """OpenAI chat-completion client with Hugging Face model compatibility."""

    def __init__(self, model: str) -> None:
        from openai import AsyncOpenAI
        self._BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        self._MODEL = model
        self._API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        self._client = AsyncOpenAI(
            base_url=self._BASE_URL,
            api_key=self._API_KEY
        )

    async def complete(self, system: str, user: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self._MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if not callable(close):
            return

        result = close()
        if asyncio.iscoroutine(result):
            await result

# ── Logging ────────────────────────────────────────────────────────────────────

def log_start(task: str, adapter: str) -> None:
    logger.info(f"[START] task={task} adapter={adapter}")


def log_step(step: int, action: any, reward: float, total: float, done: bool) -> None:
    logger.info(
        f"[STEP]  step={step} action={action} reward={reward:.2f} total={total:.1f} done={done}"
    )

def log_end(steps: int, total_reward: float, score: float) -> None:
    logger.info(
        f"[END]   steps={steps} total_reward={total_reward:.1f} score={score:.2f}"
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
    llm: HuggingFaceClient,
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

    if IMAGE_NAME is not None:
        logger.info(f"Using Docker image: {IMAGE_NAME}")
        env = await AccordisEnvironment.from_docker_image(image=IMAGE_NAME)
    else:
        raise ValueError("IMAGE_NAME environment variable must be set to run the inference script.")

    llm: Optional[HuggingFaceClient] = None
    if provider != "static":
        llm = HuggingFaceClient(model=model)

    log_start(task_name, os.getenv("ACCORDIS_ADAPTER", "simulated"))

    total_reward = 0.0
    steps_taken  = 0
    score        = 0.0

    try:
        reset_result = await env.reset(**conds)
        obs: MultiNodeObservation = reset_result.observation
        last_reward  = 0.0

        episode_max_steps = conds.get("max_steps", int(os.getenv("ACCORDIS_MAX_STEPS", "100")))
        for step in range(1, episode_max_steps + 1):
            if provider == "static":
                action = _get_static_action(obs)
            else:
                action = await _get_llm_action(llm, step, obs, last_reward)
            
            step_result = await env.step(action)

            obs = step_result.observation
            reward       = float(step_result.reward) if step_result.reward is not None else 0.0
            done         = bool(step_result.done)
            last_reward  = reward
            total_reward += reward
            steps_taken   = step

            log_step(step=step, action=action, reward=reward, total=total_reward, done=done)

            if done:
                break

        state = await env.state()
        if state.episode_log is not None:
            score = task.grade(state.episode_log)

    except Exception as exc:
        logger.error(f"[ERROR] Episode error: {exc}", exc_info=True)

    finally:
        if llm is not None:
            try:
                await llm.close()
            except Exception:
                pass
        try:
            await env.close()
        except Exception as e:
            logger.error(f"[DEBUG] env.close() error (container cleanup): {e}")
        log_end(steps=steps_taken, total_reward=total_reward, score=score)

    return {
        "task":         task_name,
        "steps":        steps_taken,
        "total_reward": round(total_reward, 2),
        "score":        round(score, 4),
    }


async def inference(
    provider: str,
    model: Optional[str],
    tasks: Optional[list[str]] = None,
) -> dict:
    """Run baseline evaluation across one or more task difficulties.

    Args:
        tasks:    List of task names to run. Defaults to all three if None.
        provider: Inference provider — "static" or "openai".
        model:    LLM model name (required when provider is not "static").

    Returns:
        A dict with status and per-task results.
    """
    if tasks is None:
        tasks = ["easy", "medium", "hard"]

    results = {}
    for task_name in tasks:
        results[task_name] = await _run_single_task(task_name, provider, model)

    inference_result = {
        "status":   "success",
        "provider": provider,
        "tasks":    tasks,
        "data":     results,
    }
    logger.info(f"Baseline result: {json.dumps(inference_result, indent=2)}")
    return inference_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Accordis baseline evaluation")
    parser.add_argument(
        "--provider", default="huggingface",
        choices=["static", "huggingface"],
        help="LLM provider (default: huggingface)"
    )
    parser.add_argument(
        "--model", default=None,
        help="LLM model to use (default: provider-specific default)"
    )
    parser.add_argument(
        "--tasks", nargs="*", default=os.getenv("ACCORDIS_TASKS", "easy,medium,hard").split(","),
        help="Difficulty levels to evaluate (default: all(easy,medium,hard). Ignored when --scenario is set."
    )
    args = parser.parse_args()
    if args.provider == "huggingface" and args.model is None:
        args.model = os.getenv("MODEL", "Qwen/Qwen2.5-72B-Instruct")
    
    asyncio.run(inference(
        provider=args.provider,
        model=args.model,
        tasks=args.tasks
    ))
