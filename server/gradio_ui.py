"""Custom Gradio UI for Accordis.

Adds an environment-specific dashboard beside the default OpenEnv playground.
"""

from __future__ import annotations

import html
import json
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

import gradio as gr


CONFIG_HEADERS = [
    "node_id",
    "view_timeout_ms",
    "pipeline_depth",
    "replication_batch_size",
    "equivocation_threshold",
    "vote_aggregation_timeout_ms",
]


def build_accordis_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """Build the Accordis-specific Gradio dashboard."""
    del action_fields, is_chat_env

    initial_snapshot = {"last_response": None, "last_operation": "reset"}

    async def handle_reset(
        n_nodes: int,
        f_byzantine: int,
        curriculum_level: int,
        max_steps: int,
        pool_size: int,
        leader_rotation: str,
        session_snapshot: Dict[str, Any],
    ):
        del session_snapshot
        honest_floor = max(0, int(n_nodes) - 1)
        bounded_f_byzantine = max(0, min(int(f_byzantine), honest_floor))
        response = await web_manager.reset_environment(
            {
                "n_nodes": int(n_nodes),
                "f_byzantine": bounded_f_byzantine,
                "curriculum_level": int(curriculum_level),
                "max_steps": int(max_steps),
                "pool_size": int(pool_size),
                "leader_rotation": str(leader_rotation),
            }
        )
        session = {"last_response": response, "last_operation": "reset"}
        state = _safe_get_state(web_manager)
        dashboard = _render_dashboard(state, response, "reset")
        status = (
            f"Reset complete. Episode {dashboard['state_summary'].get('episode_id', 'n/a')} "
            f"started with {dashboard['state_summary'].get('honest_nodes', 0)} honest nodes."
        )
        return (
            dashboard["operation_html"],
            dashboard["summary_html"],
            dashboard["graph_html"],
            dashboard["action_rows"],
            dashboard["state_json"],
            dashboard["response_json"],
            status,
            session,
        )

    async def handle_step(
        action_rows: Any,
        session_snapshot: Dict[str, Any],
    ):
        response = await web_manager.step_environment(_action_from_rows(action_rows))
        session = {"last_response": response, "last_operation": "step"}
        state = _safe_get_state(web_manager)
        dashboard = _render_dashboard(state, response, "step")
        reward = response.get("reward")
        done = response.get("done")
        reward_text = "n/a" if reward is None else f"{float(reward):.3f}"
        status = f"Step complete. Reward {reward_text}. Done: {bool(done)}."
        del session_snapshot
        return (
            dashboard["operation_html"],
            dashboard["summary_html"],
            dashboard["graph_html"],
            dashboard["action_rows"],
            dashboard["state_json"],
            dashboard["response_json"],
            status,
            session,
        )

    def handle_state(session_snapshot: Dict[str, Any]):
        last_response = (session_snapshot or {}).get("last_response")
        dashboard = _render_dashboard(_safe_get_state(web_manager), last_response, "get_state")
        session = {
            "last_response": last_response,
            "last_operation": "get_state",
        }
        status = "State snapshot refreshed from get_state()."
        return (
            dashboard["operation_html"],
            dashboard["summary_html"],
            dashboard["graph_html"],
            dashboard["action_rows"],
            dashboard["state_json"],
            dashboard["response_json"],
            status,
            session,
        )

    def handle_load(session_snapshot: Dict[str, Any]):
        current_state = _safe_get_state(web_manager, allow_empty=True)
        last_response = (session_snapshot or {}).get("last_response")
        last_operation = (session_snapshot or {}).get("last_operation") or "reset"
        dashboard = _render_dashboard(current_state, last_response, last_operation)
        return (
            dashboard["operation_html"],
            dashboard["summary_html"],
            dashboard["graph_html"],
            dashboard["action_rows"],
            dashboard["state_json"],
            dashboard["response_json"],
            "Use reset() to start a cluster, then step() or get_state() to inspect it.",
            session_snapshot or initial_snapshot,
        )

    with gr.Blocks(title=title) as demo:
        gr.HTML(_dashboard_styles())
        session_state = gr.State(initial_snapshot)

        gr.Markdown(
            f"""
            # Accordis Control Center

            Visualize the consensus cluster as a live node topology, hover nodes for details,
            and drive the environment through `reset()`, `step()`, and `get_state()`.
            """
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### API Controls")
                n_nodes = gr.Slider(
                    minimum=4,
                    maximum=12,
                    step=1,
                    value=4,
                    label="n_nodes",
                )
                f_byzantine = gr.Slider(
                    minimum=0,
                    maximum=4,
                    step=1,
                    value=0,
                    label="f_byzantine",
                )
                curriculum_level = gr.Slider(
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=1,
                    label="curriculum_level",
                )
                max_steps = gr.Slider(
                    minimum=10,
                    maximum=200,
                    step=10,
                    value=200,
                    label="max_steps",
                )
                pool_size = gr.Slider(
                    minimum=100,
                    maximum=2000,
                    step=100,
                    value=1000,
                    label="pool_size",
                )
                leader_rotation = gr.Dropdown(
                    choices=["round_robin", "vrf", "reputation_weighted"],
                    value="round_robin",
                    label="leader_rotation",
                )
                with gr.Row():
                    reset_btn = gr.Button("reset()", variant="primary")
                    state_btn = gr.Button("get_state()", variant="secondary")

                gr.Markdown("### step() Action Table")
                action_table = gr.Dataframe(
                    headers=CONFIG_HEADERS,
                    datatype=["str", "number", "number", "number", "number", "number"],
                    value=[],
                    interactive=True,
                    wrap=True,
                    row_count=(1, "dynamic"),
                    col_count=(len(CONFIG_HEADERS), "fixed"),
                    label="Editable per-node BFT configuration",
                )
                step_btn = gr.Button("step()", variant="secondary")
                status_box = gr.Markdown(
                    "Use reset() to populate the table with honest nodes."
                )

                if quick_start_md:
                    with gr.Accordion("Quick Start", open=False):
                        gr.Markdown(quick_start_md)
                readme_content = getattr(metadata, "readme_content", "") if metadata else ""
                if readme_content:
                    with gr.Accordion("README", open=False):
                        gr.Markdown(readme_content)

            with gr.Column(scale=2):
                operation_html = gr.HTML(_empty_operation_html())
                summary_html = gr.HTML(_empty_summary_html())
                graph_html = gr.HTML(_empty_graph_html())

        with gr.Row():
            state_json = gr.Code(
                label="State Snapshot",
                value=_pretty_json({"message": "State will appear here after get_state() or reset()."}),
                language="json",
                interactive=False,
            )
            response_json = gr.Code(
                label="Last API Response",
                value=_pretty_json({"message": "The latest reset() or step() payload will appear here."}),
                language="json",
                interactive=False,
            )

        outputs = [
            operation_html,
            summary_html,
            graph_html,
            action_table,
            state_json,
            response_json,
            status_box,
            session_state,
        ]

        demo.load(fn=handle_load, inputs=[session_state], outputs=outputs)
        reset_btn.click(
            fn=handle_reset,
            inputs=[
                n_nodes,
                f_byzantine,
                curriculum_level,
                max_steps,
                pool_size,
                leader_rotation,
                session_state,
            ],
            outputs=outputs,
        )
        step_btn.click(
            fn=handle_step,
            inputs=[action_table, session_state],
            outputs=outputs,
        )
        state_btn.click(
            fn=handle_state,
            inputs=[session_state],
            outputs=outputs,
        )

    return demo


def _safe_get_state(web_manager: Any, allow_empty: bool = False) -> Dict[str, Any]:
    state = _json_safe(web_manager.get_state())
    if allow_empty or state.get("node_states"):
        return state
    return {}


def _render_dashboard(
    state: Optional[Dict[str, Any]],
    response: Optional[Dict[str, Any]],
    last_operation: str,
) -> Dict[str, Any]:
    safe_state = _json_safe(state or {})
    safe_response = _json_safe(response or {})
    state_summary = _compact_state_for_display(safe_state, safe_response)
    return {
        "operation_html": _build_operation_html(state_summary, safe_response, last_operation),
        "summary_html": _build_summary_html(state_summary, safe_response),
        "graph_html": _build_node_graph_html(safe_state, safe_response.get("observation", {})),
        "action_rows": _action_rows_from_state(safe_state, safe_response.get("observation", {})),
        "state_json": _pretty_json(state_summary),
        "response_json": _pretty_json(safe_response or {"message": "No reset() or step() response yet."}),
        "state_summary": state_summary,
    }


def _compact_state_for_display(
    state: Dict[str, Any],
    response: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    node_states = state.get("node_states", {}) or {}
    observation_nodes = ((response or {}).get("observation") or {}).get("nodes", {})

    node_summaries = []
    honest_nodes = 0
    byzantine_nodes = 0
    pending_count = 0

    for node_id, node_state in sorted(node_states.items()):
        is_byzantine = bool(node_state.get("is_byzantine"))
        observation = observation_nodes.get(node_id, {})
        config = node_state.get("config", {}) or observation.get("current_config", {})
        committed_blocks = len(node_state.get("committed_log", []) or [])
        role = observation.get("current_role") or node_state.get("current_role") or "replica"
        view = observation.get("current_view", node_state.get("current_view", 0))
        pending_count = max(pending_count, int(observation.get("pending_txn_count", 0) or 0))
        node_summaries.append(
            {
                "node_id": node_id,
                "is_byzantine": is_byzantine,
                "role": role,
                "view": int(view or 0),
                "committed_blocks": committed_blocks,
                "throughput_tps": float(observation.get("commit_throughput_tps", 0.0) or 0.0),
                "pipeline_utilisation": float(observation.get("pipeline_utilisation", 0.0) or 0.0),
                "pending_txn_count": int(observation.get("pending_txn_count", 0) or 0),
                "config": config,
            }
        )
        if is_byzantine:
            byzantine_nodes += 1
        else:
            honest_nodes += 1

    return {
        "episode_id": state.get("episode_id"),
        "step": int(state.get("step", 0) or 0),
        "curriculum_level": int(state.get("curriculum_level", 0) or 0),
        "leader_rotation": state.get("leader_rotation"),
        "bfa_strategy": state.get("bfa_strategy"),
        "n_nodes": int(state.get("n_nodes", len(node_states)) or len(node_states)),
        "f_byzantine": int(state.get("f_byzantine", byzantine_nodes) or byzantine_nodes),
        "honest_nodes": honest_nodes,
        "byzantine_nodes": byzantine_nodes,
        "view_change_count": int(state.get("view_change_count", 0) or 0),
        "finalized_txn_count": int(state.get("finalized_txn_count", 0) or 0),
        "txn_pool_size": len(state.get("episode_txn_pool", []) or []),
        "proposal_registry_size": len(
            ((state.get("proposal_registry") or {}).get("honest_proposals") or {})
        ),
        "pending_txn_count": pending_count,
        "nodes": node_summaries,
    }


def _build_operation_html(
    state_summary: Dict[str, Any],
    response: Dict[str, Any],
    last_operation: str,
) -> str:
    reward = response.get("reward")
    reward_value = "n/a" if reward is None else f"{float(reward):.3f}"
    done = bool(response.get("done")) if response else False
    termination = (response.get("metadata") or {}).get("termination_reason") or "active"
    episode_id = state_summary.get("episode_id") or "not-started"
    cards = [
        (
            "reset()",
            "Create a fresh cluster, configure network conditions, and seed the episode.",
            f"{state_summary.get('honest_nodes', 0)} honest / {state_summary.get('byzantine_nodes', 0)} Byzantine",
        ),
        (
            "step()",
            "Apply the per-node tuning table, advance one consensus round, and capture reward.",
            f"step {state_summary.get('step', 0)} | reward {reward_value} | done {str(done).lower()}",
        ),
        (
            "get_state()",
            "Inspect the authoritative environment state without advancing the simulation.",
            f"episode {episode_id} | termination {termination}",
        ),
    ]

    card_html = []
    for name, description, live_detail in cards:
        active_class = " api-card-active" if name.rstrip("()") == last_operation.rstrip("()") else ""
        card_html.append(
            f"""
            <div class="api-card{active_class}">
              <div class="api-card-title">{html.escape(name)}</div>
              <div class="api-card-copy">{html.escape(description)}</div>
              <div class="api-card-detail">{html.escape(live_detail)}</div>
            </div>
            """
        )

    return f"""
    <div class="accordis-shell">
      <div class="section-title">API Flow</div>
      <div class="api-grid">
        {''.join(card_html)}
      </div>
    </div>
    """


def _build_summary_html(state_summary: Dict[str, Any], response: Dict[str, Any]) -> str:
    reward = response.get("reward")
    stats = [
        ("Honest Nodes", str(state_summary.get("honest_nodes", 0))),
        ("Byzantine Nodes", str(state_summary.get("byzantine_nodes", 0))),
        ("Finalized Txns", str(state_summary.get("finalized_txn_count", 0))),
        ("Pending Txns", str(state_summary.get("pending_txn_count", 0))),
        ("View Changes", str(state_summary.get("view_change_count", 0))),
        (
            "Reward",
            "n/a" if reward is None else f"{float(reward):.3f}",
        ),
    ]
    stat_html = [
        f"""
        <div class="metric-card">
          <div class="metric-label">{html.escape(label)}</div>
          <div class="metric-value">{html.escape(value)}</div>
        </div>
        """
        for label, value in stats
    ]
    return f"""
    <div class="accordis-shell">
      <div class="section-title">Live Metrics</div>
      <div class="metric-grid">
        {''.join(stat_html)}
      </div>
    </div>
    """


def _build_node_graph_html(
    state: Dict[str, Any],
    observation: Optional[Dict[str, Any]] = None,
) -> str:
    node_states = sorted((state.get("node_states") or {}).items())
    observation_nodes = (observation or {}).get("nodes", {}) if observation else {}
    if not node_states:
        return _empty_graph_html()

    width = 880
    height = 520
    center_x = width / 2
    center_y = height / 2
    radius = 175 if len(node_states) <= 6 else 205

    node_svg = []
    edge_svg = []

    for index, (node_id, node_state) in enumerate(node_states):
        angle = (2 * math.pi * index / max(1, len(node_states))) - (math.pi / 2)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        obs = observation_nodes.get(node_id, {})
        role = obs.get("current_role") or node_state.get("current_role") or "replica"
        is_byzantine = bool(node_state.get("is_byzantine"))
        leader = role == "leader"
        fill = "#ff8d6b" if is_byzantine else "#3cc0b1"
        stroke = "#5b2d1f" if is_byzantine else "#0f4f4b"
        role_label = "BYZ" if is_byzantine else str(role).upper()
        config = node_state.get("config", {}) or {}
        tooltip = "\n".join(
            [
                node_id,
                f"role: {role}",
                f"view: {obs.get('current_view', node_state.get('current_view', 0))}",
                f"committed blocks: {len(node_state.get('committed_log', []) or [])}",
                f"pending txns: {obs.get('pending_txn_count', 0)}",
                f"throughput: {float(obs.get('commit_throughput_tps', 0.0) or 0.0):.2f} TPS",
                f"pipeline util: {float(obs.get('pipeline_utilisation', 0.0) or 0.0):.2f}",
                f"timeout: {config.get('view_timeout_ms', 0)} ms",
                f"batch: {config.get('replication_batch_size', 0)}",
            ]
        )
        edge_svg.append(
            f'<line x1="{center_x}" y1="{center_y}" x2="{x}" y2="{y}" '
            'stroke="rgba(38, 75, 88, 0.18)" stroke-width="2" />'
        )
        ring = (
            f'<circle cx="{x}" cy="{y}" r="32" fill="none" '
            'stroke="#f4c95d" stroke-width="3" />'
            if leader
            else ""
        )
        node_svg.append(
            f"""
            <g>
              <title>{html.escape(tooltip)}</title>
              {ring}
              <circle cx="{x}" cy="{y}" r="24" fill="{fill}" stroke="{stroke}" stroke-width="2.5" />
              <text x="{x}" y="{y - 4}" text-anchor="middle" class="node-name">{html.escape(node_id)}</text>
              <text x="{x}" y="{y + 14}" text-anchor="middle" class="node-role">{html.escape(role_label)}</text>
            </g>
            """
        )

    leader_rotation = state.get("leader_rotation", "unknown")
    strategy = state.get("bfa_strategy", "none")
    step = state.get("step", 0)
    return f"""
    <div class="accordis-shell">
      <div class="section-title">Consensus Topology</div>
      <div class="graph-caption">
        Hover nodes to inspect role, view, throughput, pending transactions, and config.
      </div>
      <svg class="topology-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Accordis node graph">
        <defs>
          <radialGradient id="accordisCore" cx="50%" cy="50%" r="70%">
            <stop offset="0%" stop-color="#effcf9"></stop>
            <stop offset="100%" stop-color="#dbeef1"></stop>
          </radialGradient>
        </defs>
        <rect x="0" y="0" width="{width}" height="{height}" rx="28" fill="#f7fbfc"></rect>
        <circle cx="{center_x}" cy="{center_y}" r="82" fill="url(#accordisCore)" stroke="#9fc7cf" stroke-width="2"></circle>
        <text x="{center_x}" y="{center_y - 10}" text-anchor="middle" class="core-title">Consensus Fabric</text>
        <text x="{center_x}" y="{center_y + 14}" text-anchor="middle" class="core-copy">step {html.escape(str(step))}</text>
        <text x="{center_x}" y="{center_y + 34}" text-anchor="middle" class="core-copy">
          {html.escape(str(leader_rotation))} | {html.escape(str(strategy))}
        </text>
        {''.join(edge_svg)}
        {''.join(node_svg)}
      </svg>
    </div>
    """


def _action_rows_from_state(
    state: Dict[str, Any],
    observation: Optional[Dict[str, Any]] = None,
) -> List[List[Any]]:
    rows: List[List[Any]] = []
    node_states = state.get("node_states", {}) or {}
    observation_nodes = (observation or {}).get("nodes", {}) if observation else {}
    for node_id, node_state in sorted(node_states.items()):
        if node_state.get("is_byzantine"):
            continue
        config = node_state.get("config", {}) or (
            (observation_nodes.get(node_id) or {}).get("current_config", {})
        )
        rows.append(
            [
                node_id,
                int(config.get("view_timeout_ms", 2000) or 2000),
                int(config.get("pipeline_depth", 2) or 2),
                int(config.get("replication_batch_size", 64) or 64),
                int(config.get("equivocation_threshold", 5) or 5),
                int(config.get("vote_aggregation_timeout_ms", 500) or 500),
            ]
        )
    return rows


def _action_from_rows(rows: Any) -> Dict[str, Any]:
    normalized_rows = _normalize_rows(rows)
    nodes: Dict[str, Dict[str, Any]] = {}
    for row in normalized_rows:
        if not row:
            continue
        node_id = str(row[0]).strip()
        if not node_id:
            continue
        nodes[node_id] = {
            "node_id": node_id,
            "view_timeout_ms": _as_int(row, 1, 2000),
            "pipeline_depth": _as_int(row, 2, 2),
            "replication_batch_size": _as_int(row, 3, 64),
            "equivocation_threshold": _as_int(row, 4, 5),
            "vote_aggregation_timeout_ms": _as_int(row, 5, 500),
        }
    if not nodes:
        raise gr.Error("Reset the environment first, then edit the action table before calling step().")
    return {"nodes": nodes}


def _normalize_rows(rows: Any) -> List[List[Any]]:
    if rows is None:
        return []
    if hasattr(rows, "values") and hasattr(rows.values, "tolist"):
        return rows.values.tolist()
    if isinstance(rows, list):
        return rows
    return []


def _as_int(row: Sequence[Any], index: int, default: int) -> int:
    try:
        value = row[index]
    except IndexError:
        return default
    if value in (None, ""):
        return default
    return int(float(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _pretty_json(value: Any) -> str:
    return json.dumps(_json_safe(value), indent=2, sort_keys=False)


def _empty_operation_html() -> str:
    return _build_operation_html({}, {}, "reset")


def _empty_summary_html() -> str:
    return _build_summary_html({}, {})


def _empty_graph_html() -> str:
    return """
    <div class="accordis-shell">
      <div class="section-title">Consensus Topology</div>
      <div class="graph-empty">
        Reset the environment to render the node topology.
      </div>
    </div>
    """


def _dashboard_styles() -> str:
    return """
    <style>
      .accordis-shell {
        --bg-soft: #f7fbfc;
        --bg-panel: #ffffff;
        --ink: #18323d;
        --muted: #5d7983;
        --teal: #3cc0b1;
        --amber: #f4c95d;
        --coral: #ff8d6b;
        --line: #d5e5e8;
        font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
        color: var(--ink);
      }

      .section-title {
        font-size: 1.02rem;
        font-weight: 700;
        letter-spacing: 0.01em;
        margin-bottom: 0.6rem;
      }

      .api-grid,
      .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
      }

      .api-card,
      .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbfc 100%);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem;
        box-shadow: 0 12px 28px rgba(17, 41, 51, 0.06);
      }

      .api-card-active {
        border-color: #7fcfc5;
        box-shadow: 0 16px 30px rgba(60, 192, 177, 0.16);
      }

      .api-card-title,
      .metric-label {
        color: var(--muted);
        font-size: 0.88rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
      }

      .api-card-copy {
        font-size: 0.94rem;
        line-height: 1.45;
        min-height: 3.6rem;
      }

      .api-card-detail,
      .metric-value {
        margin-top: 0.7rem;
        font-size: 1.05rem;
        font-weight: 700;
      }

      .graph-caption {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 0.75rem;
      }

      .graph-empty {
        background: linear-gradient(180deg, #ffffff 0%, #f7fbfc 100%);
        border: 1px dashed var(--line);
        border-radius: 22px;
        padding: 2.4rem 1.2rem;
        text-align: center;
        color: var(--muted);
      }

      .topology-svg {
        width: 100%;
        height: auto;
        border-radius: 24px;
        border: 1px solid var(--line);
        box-shadow: 0 18px 36px rgba(17, 41, 51, 0.07);
        background: var(--bg-soft);
      }

      .node-name {
        fill: #0d2c38;
        font-size: 11px;
        font-weight: 700;
      }

      .node-role {
        fill: #123f49;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.08em;
      }

      .core-title {
        fill: #204553;
        font-size: 20px;
        font-weight: 700;
      }

      .core-copy {
        fill: #50707c;
        font-size: 12px;
        font-weight: 600;
      }

      @media (max-width: 980px) {
        .api-grid,
        .metric-grid {
          grid-template-columns: 1fr;
        }
      }
    </style>
    """
