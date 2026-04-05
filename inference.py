"""
Accordis Inference Script
===================================
MANDATORY environment variables:
    ACCORDIS_ADAPTER   Adapter to use: "simulated" (default) or "librabft".

LLM selection (LLMClientFactory, priority order):
    OPENAI_API_KEY     → OpenAI gpt-4o
    GEMINI_API_KEY     → Google gemini-1.5-pro

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

import json
import os
import asyncio
import argparse
from dotenv import load_dotenv

from accordis.server.utils.baseline_helper import run_baseline
from accordis.server.utils.logger import logger

async def inference(provider: str, model: str, tasks: list[str]) -> dict:
    baseline_result = await run_baseline(
        provider=provider,
        model=model,
        tasks=tasks
    )
    logger.info(f"Baseline result: {json.dumps(baseline_result, indent=2)}")
    return baseline_result

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Accordis baseline evaluation")
    parser.add_argument(
        "--provider", default=os.getenv("PROVIDER", "openai"),
        choices=["static", "openai", "gemini"],
        help="LLM provider (default: openai)"
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
    if args.provider in ["openai", "gemini"] and args.model is None:
        if args.provider == "openai":
            args.model = os.getenv("MODEL", "Qwen/Qwen2.5-72B-Instruct")
        elif args.provider == "gemini":
            args.model = os.getenv("MODEL", "gemini-3.1-flash-lite-preview")
    
    asyncio.run(inference(
        provider=args.provider,
        model=args.model,
        tasks=args.tasks
    ))
