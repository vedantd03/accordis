"""Custom Gradio UI for the Accordis BFT consensus environment.

Builds a "Visualizer" tab shown alongside the default OpenEnv Playground tab.
The tab contains:
  - A Plotly network graph of all nodes (honest + Byzantine), with hover tooltips.
  - A live activity log showing the last operation (reset / step / get_state).
  - Three action buttons that call reset(), step(), and get_state() via web_manager.
  - A real-time metrics panel (throughput, view-changes, BFA strategy).
"""

from __future__ import annotations

import json
import math
import asyncio
from typing import Any, Dict, List, Optional

import gradio as gr

# ---------------------------------------------------------------------------
# Plotly is an optional dependency; we surface a clear message if missing.
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:  # pragma: no cover
    _PLOTLY_OK = False


# ── Colour palette (dark, modern) ──────────────────────────────────────────
_DARK_BG       = "#0f1117"
_CARD_BG       = "#1a1d27"
_BORDER        = "#2a2d3e"
_ACCENT_BLUE   = "#4f8ef7"
_ACCENT_GREEN  = "#22c55e"
_ACCENT_RED    = "#ef4444"
_ACCENT_AMBER  = "#f59e0b"
_ACCENT_PURPLE = "#a855f7"
_TEXT_PRIMARY  = "#e2e8f0"
_TEXT_MUTED    = "#64748b"
_LEADER_COLOUR   = _ACCENT_BLUE
_REPLICA_COLOUR  = _ACCENT_GREEN
_BYZANTINE_COLOUR = _ACCENT_RED
_CANDIDATE_COLOUR = _ACCENT_AMBER

_CSS = f"""
/* ── Root / global ────────────────────────────────────────────── */
body, .gradio-container {{
    background: {_DARK_BG} !important;
    color: {_TEXT_PRIMARY} !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}}
/* ── Panel cards ─────────────────────────────────────────────── */
.accordis-card {{
    background: {_CARD_BG};
    border: 1px solid {_BORDER};
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}}
/* ── Section titles ──────────────────────────────────────────── */
.accordis-section-title {{
    color: {_TEXT_MUTED};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}}
/* ── Metric chips ────────────────────────────────────────────── */
.metric-row {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}}
.metric-chip {{
    background: {_DARK_BG};
    border: 1px solid {_BORDER};
    border-radius: 8px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 500;
}}
.metric-chip .label {{ color: {_TEXT_MUTED}; font-size: 11px; display: block; }}
.metric-chip .value {{ color: {_TEXT_PRIMARY}; font-size: 15px; }}
/* ── Log panel ───────────────────────────────────────────────── */
.log-panel {{
    background: {_DARK_BG};
    border: 1px solid {_BORDER};
    border-radius: 8px;
    padding: 12px 14px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    color: {_TEXT_PRIMARY};
    max-height: 220px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}}
/* ── Buttons ─────────────────────────────────────────────────── */
.btn-reset  {{ background: {_ACCENT_PURPLE} !important; color: #fff !important; border: none !important; border-radius: 8px !important; }}
.btn-step   {{ background: {_ACCENT_BLUE}   !important; color: #fff !important; border: none !important; border-radius: 8px !important; }}
.btn-state  {{ background: {_CARD_BG}       !important; color: {_TEXT_PRIMARY} !important; border: 1px solid {_BORDER} !important; border-radius: 8px !important; }}
/* ── Legend dots ─────────────────────────────────────────────── */
.legend {{ display: flex; gap: 18px; flex-wrap: wrap; font-size: 12px; color: {_TEXT_MUTED}; margin-top: 4px; }}
.legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
"""


# ── Graph builder ──────────────────────────────────────────────────────────

def _ring_positions(n: int) -> List[tuple[float, float]]:
    """Evenly-spaced positions on a unit circle."""
    return [
        (math.cos(2 * math.pi * i / n - math.pi / 2),
         math.sin(2 * math.pi * i / n - math.pi / 2))
        for i in range(n)
    ]


def _build_node_graph(state: Optional[Dict[str, Any]]) -> "go.Figure":
    """Return a Plotly figure from parsed state dict (or a placeholder)."""
    if not _PLOTLY_OK:
        fig = go.Figure()
        fig.add_annotation(text="plotly not installed", x=0.5, y=0.5, showarrow=False)
        return fig

    # ── Parse node list from state ────────────────────────────────────────
    node_states: Dict[str, Any] = {}
    bfa_strategy = "none"
    step_num = 0
    curriculum = 1
    n_nodes = 0
    f_byzantine = 0

    if state:
        node_states  = state.get("node_states", {})
        bfa_strategy = state.get("bfa_strategy", "none")
        step_num     = state.get("step", 0)
        curriculum   = state.get("curriculum_level", 1)
        n_nodes      = state.get("n_nodes", len(node_states))
        f_byzantine  = state.get("f_byzantine", 0)

    if not node_states:
        # Placeholder: show an instructional message inside the figure
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=_CARD_BG,
            plot_bgcolor=_CARD_BG,
            font=dict(color=_TEXT_MUTED),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            annotations=[dict(
                text="Click <b>Reset</b> to initialise the cluster",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=15, color=_TEXT_MUTED),
            )],
        )
        return fig

    node_ids  = list(node_states.keys())
    positions = _ring_positions(len(node_ids))

    # ── Edge traces (full mesh) ────────────────────────────────────────────
    edge_x, edge_y = [], []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            x0, y0 = positions[i]
            x1, y1 = positions[j]
            ns_i = node_states[node_ids[i]]
            ns_j = node_states[node_ids[j]]
            # Dim edges involving Byzantine nodes
            if ns_i.get("is_byzantine") or ns_j.get("is_byzantine"):
                continue  # draw Byzantine edges separately below
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    byz_ex, byz_ey = [], []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            ns_i = node_states[node_ids[i]]
            ns_j = node_states[node_ids[j]]
            if ns_i.get("is_byzantine") or ns_j.get("is_byzantine"):
                x0, y0 = positions[i]
                x1, y1 = positions[j]
                byz_ex += [x0, x1, None]
                byz_ey += [y0, y1, None]

    traces = []
    if edge_x:
        traces.append(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=1.2, color="rgba(100,116,139,0.3)"),
            hoverinfo="none", showlegend=False,
        ))
    if byz_ex:
        traces.append(go.Scatter(
            x=byz_ex, y=byz_ey, mode="lines",
            line=dict(width=1, color="rgba(239,68,68,0.2)", dash="dot"),
            hoverinfo="none", showlegend=False,
        ))

    # ── Node traces ────────────────────────────────────────────────────────
    node_x, node_y, node_colours, node_sizes = [], [], [], []
    node_symbols: List[str] = []
    hover_texts: List[str] = []
    node_labels: List[str] = []

    for idx, nid in enumerate(node_ids):
        ns   = node_states[nid]
        x, y = positions[idx]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(nid)

        is_byz  = ns.get("is_byzantine", False)
        role    = ns.get("current_role", "replica")
        view    = ns.get("current_view", 0)
        cfg     = ns.get("config", {})
        log_len = len(ns.get("committed_log", []))

        if is_byz:
            colour = _BYZANTINE_COLOUR
            symbol = "x"
            size   = 22
        elif role == "leader":
            colour = _LEADER_COLOUR
            symbol = "star"
            size   = 28
        elif role == "candidate":
            colour = _CANDIDATE_COLOUR
            symbol = "diamond"
            size   = 24
        else:
            colour = _REPLICA_COLOUR
            symbol = "circle"
            size   = 22

        node_colours.append(colour)
        node_symbols.append(symbol)
        node_sizes.append(size)

        # ── Hover tooltip ────────────────────────────────────────────────
        vto  = cfg.get("view_timeout_ms",             "—")
        pd_  = cfg.get("pipeline_depth",              "—")
        rbs  = cfg.get("replication_batch_size",      "—")
        eq_t = cfg.get("equivocation_threshold",      "—")
        vat  = cfg.get("vote_aggregation_timeout_ms", "—")
        role_label = "BYZANTINE" if is_byz else role.upper()
        hover_texts.append(
            f"<b>{nid}</b><br>"
            f"Role: <b>{role_label}</b><br>"
            f"View: {view} · Log: {log_len} blocks<br>"
            f"──────────────────<br>"
            f"view_timeout_ms: {vto}<br>"
            f"pipeline_depth: {pd_}<br>"
            f"replication_batch_size: {rbs}<br>"
            f"equivocation_threshold: {eq_t}<br>"
            f"vote_agg_timeout_ms: {vat}"
        )

    traces.append(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(
            color=node_colours,
            size=node_sizes,
            symbol=node_symbols,
            line=dict(width=2, color=_CARD_BG),
        ),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=11, color=_TEXT_PRIMARY),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    ))

    # ── Layout ─────────────────────────────────────────────────────────────
    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor=_CARD_BG,
        plot_bgcolor=_CARD_BG,
        font=dict(color=_TEXT_PRIMARY, family="Inter, sans-serif"),
        xaxis=dict(visible=False, range=[-1.45, 1.45]),
        yaxis=dict(visible=False, range=[-1.45, 1.45], scaleanchor="x"),
        margin=dict(l=10, r=10, t=30, b=10),
        hoverlabel=dict(
            bgcolor=_DARK_BG,
            bordercolor=_BORDER,
            font=dict(color=_TEXT_PRIMARY, size=12),
        ),
        title=dict(
            text=(
                f"<b>Cluster</b>  ·  Step {step_num}  ·  "
                f"Level {curriculum}  ·  BFA: <i>{bfa_strategy}</i>"
            ),
            font=dict(size=13, color=_TEXT_MUTED),
            x=0.5, xanchor="center",
        ),
    )
    return fig


# ── Metrics formatter ──────────────────────────────────────────────────────

def _metrics_html(state: Optional[Dict[str, Any]], obs: Optional[Dict[str, Any]]) -> str:
    """Return an HTML snippet with metric chips (all inline styles — no CSS class dependency)."""
    _chip_base = (
        f"display:inline-flex;flex-direction:column;gap:2px;"
        f"background:{_DARK_BG};border:1px solid {_BORDER};"
        f"border-radius:8px;padding:8px 14px;min-width:80px;"
    )
    _label_style = f"font-size:10px;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;color:{_TEXT_MUTED};"
    _section_style = (
        f"font-size:11px;font-weight:600;letter-spacing:0.08em;"
        f"text-transform:uppercase;color:{_TEXT_MUTED};"
        f"margin-bottom:10px;"
    )

    if not state:
        return (
            f'<div style="{_section_style}">Metrics</div>'
            f'<p style="color:{_TEXT_MUTED};font-size:13px;margin:0">No data — click Reset first.</p>'
        )

    step_num    = state.get("step", 0)
    curriculum  = state.get("curriculum_level", 1)
    n_nodes     = state.get("n_nodes", 0)
    f_byzantine = state.get("f_byzantine", 0)
    bfa         = str(state.get("bfa_strategy", "none")).replace("BFAStrategy.", "")
    vc          = state.get("view_change_count", 0)
    fin_txns    = state.get("finalized_txn_count", 0)

    # Aggregate throughput from per-node observations
    avg_tps = 0.0
    nodes_obs: Dict[str, Any] = {}
    if obs and isinstance(obs.get("nodes"), dict):
        nodes_obs = obs["nodes"]
    elif obs and isinstance(obs.get("observation"), dict):
        nodes_obs = obs["observation"].get("nodes", {})
    if nodes_obs:
        tps_vals = [n.get("commit_throughput_tps", 0.0) for n in nodes_obs.values()]
        avg_tps = sum(tps_vals) / max(len(tps_vals), 1)

    def chip(label: str, value: str, colour: str = _TEXT_PRIMARY) -> str:
        return (
            f'<div style="{_chip_base}">'
            f'<span style="{_label_style}">{label}</span>'
            f'<span style="font-size:16px;font-weight:700;color:{colour};line-height:1.2">{value}</span>'
            f'</div>'
        )

    bfa_colour = _ACCENT_RED if bfa not in ("none", "NONE") else _ACCENT_GREEN
    byz_colour = _ACCENT_RED if f_byzantine else _ACCENT_GREEN
    vc_colour  = _ACCENT_AMBER if vc else _TEXT_PRIMARY

    row_style = f"display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;"

    chips = "".join([
        chip("Step",           str(step_num)),
        chip("Curriculum",     str(curriculum)),
        chip("Nodes",          str(n_nodes)),
        chip("Byzantine",      str(f_byzantine), byz_colour),
        chip("BFA Strategy",   bfa,              bfa_colour),
        chip("View Changes",   str(vc),          vc_colour),
        chip("Finalized Txns", str(fin_txns),    _ACCENT_BLUE),
        chip("Avg TPS",        f"{avg_tps:.2f}", _ACCENT_BLUE),
    ])

    return (
        f'<div style="{_section_style}">Live Metrics</div>'
        f'<div style="{row_style}">{chips}</div>'
    )


# ── Log formatter ──────────────────────────────────────────────────────────

def _fmt_log(operation: str, data: Any) -> str:
    """Return a concise text summary for the activity log."""
    if isinstance(data, str):
        return f"[{operation}] ERROR — {data}"
    if not data:
        return f"[{operation}] (no data)"

    lines = [f"[{operation}]"]

    if operation == "reset":
        obs = data.get("observation", data)
        nodes = obs.get("nodes", {}) if isinstance(obs, dict) else {}
        lines.append(f"  nodes initialised: {list(nodes.keys()) or '—'}")
        lines.append(f"  done={data.get('done')}  reward={data.get('reward')}")

    elif operation == "step":
        obs = data.get("observation", data)
        nodes = obs.get("nodes", {}) if isinstance(obs, dict) else {}
        for nid, nobs in list(nodes.items())[:6]:
            role = nobs.get("current_role", "?")
            view = nobs.get("current_view", "?")
            tps  = nobs.get("commit_throughput_tps", 0.0)
            lines.append(f"  {nid}: role={role}  view={view}  tps={tps:.2f}")
        if len(nodes) > 6:
            lines.append(f"  … (+{len(nodes)-6} more nodes)")
        lines.append(f"  done={data.get('done')}  reward={data.get('reward')}")

    elif operation == "get_state":
        step = data.get("step", "?")
        bfa  = data.get("bfa_strategy", "?")
        fin  = data.get("finalized_txn_count", "?")
        lines.append(f"  step={step}  bfa_strategy={bfa}  finalized_txns={fin}")

    return "\n".join(lines)


# ── Main builder ───────────────────────────────────────────────────────────

def build_accordis_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """
    Returns a gr.Blocks that becomes the 'Custom' tab in the TabbedInterface.

    Layout
    ──────
    Left column  : cluster graph (Plotly) + legend
    Right column : metrics strip + action buttons + activity log + state JSON
    """

    # Shared in-memory state across callbacks (one env per server process)
    _shared: Dict[str, Any] = {"state": None, "last_obs": None, "log": "Ready — click Reset to begin."}

    # ── Async helpers ──────────────────────────────────────────────────────

    async def _do_reset():
        try:
            data = await web_manager.reset_environment()
        except Exception as exc:
            data = str(exc)
        # Refresh state after reset
        try:
            state = web_manager.get_state()
        except Exception:
            state = None
        _shared["state"]    = state
        _shared["last_obs"] = data if isinstance(data, dict) else None
        _shared["log"]      = _fmt_log("reset", data)
        return _refresh_ui()

    async def _do_step():
        if _shared["state"] is None:
            _shared["log"] = "[step] Not started — click Reset first."
            return _refresh_ui()
        # Build a no-op action that keeps current configs unchanged
        state       = _shared["state"]
        node_states = state.get("node_states", {}) if state else {}
        nodes_action: Dict[str, Any] = {}
        for nid, ns in node_states.items():
            if ns.get("is_byzantine"):
                continue
            cfg = ns.get("config", {})
            nodes_action[nid] = {
                "node_id":                     nid,
                "view_timeout_ms":             cfg.get("view_timeout_ms",             2000),
                "pipeline_depth":              cfg.get("pipeline_depth",              2),
                "replication_batch_size":      cfg.get("replication_batch_size",      64),
                "equivocation_threshold":      cfg.get("equivocation_threshold",      5),
                "vote_aggregation_timeout_ms": cfg.get("vote_aggregation_timeout_ms", 500),
            }
        action_payload = {"nodes": nodes_action}
        try:
            data = await web_manager.step_environment(action_payload)
        except Exception as exc:
            data = str(exc)
        try:
            new_state = web_manager.get_state()
        except Exception:
            new_state = _shared["state"]
        _shared["state"]    = new_state
        _shared["last_obs"] = data if isinstance(data, dict) else None
        _shared["log"]      = _fmt_log("step", data)
        return _refresh_ui()

    def _do_get_state():
        try:
            state = web_manager.get_state()
        except Exception as exc:
            state = {"error": str(exc)}
        _shared["state"] = state
        _shared["log"]   = _fmt_log("get_state", state)
        return _refresh_ui()

    # ── UI refresh helper (returns values for all outputs) ─────────────────

    def _refresh_ui():
        state    = _shared["state"]
        last_obs = _shared["last_obs"]
        fig      = _build_node_graph(state)
        metrics  = _metrics_html(state, last_obs)
        log_txt  = _shared["log"]
        state_json = json.dumps(state, indent=2, default=str) if state else "null"
        return fig, metrics, log_txt, state_json

    # ── Gradio layout ──────────────────────────────────────────────────────

    _section_title_style = (
        f"font-size:11px;font-weight:600;letter-spacing:0.08em;"
        f"text-transform:uppercase;color:{_TEXT_MUTED};margin-bottom:8px;"
    )

    with gr.Blocks(
        title="Accordis — BFT Cluster Visualizer",
        css=_CSS,
    ) as demo:

        gr.HTML(
            f'<div style="padding:16px 0 8px;border-bottom:1px solid {_BORDER};margin-bottom:16px">'
            f'<span style="font-size:20px;font-weight:700;color:{_TEXT_PRIMARY}">Accordis</span>'
            f'<span style="font-size:13px;color:{_TEXT_MUTED};margin-left:12px">BFT Consensus Cluster Visualizer</span>'
            f'</div>'
        )

        # ── TOP ROW: graph (left) + controls (right) ───────────────────────
        # The JSON accordion is kept OUTSIDE this row so it never shifts the graph.
        with gr.Row():

            # LEFT — cluster topology, pinned to the top with align-self
            with gr.Column(scale=3, min_width=360, elem_id="accordis-left-col"):
                gr.HTML(f'<div style="{_section_title_style}">Cluster Topology</div>')
                cluster_plot = gr.Plot(
                    value=_build_node_graph(None),
                    label="",
                    show_label=False,
                    container=False,
                )
                _legend_dot = (
                    "display:inline-block;width:10px;height:10px;"
                    "border-radius:50%;margin-right:5px;vertical-align:middle;"
                )
                gr.HTML(
                    f'<div style="display:flex;gap:18px;flex-wrap:wrap;'
                    f'font-size:12px;color:{_TEXT_MUTED};margin-top:6px;padding-bottom:4px">'
                    f'<span><span style="{_legend_dot}background:{_LEADER_COLOUR}"></span>Leader (★)</span>'
                    f'<span><span style="{_legend_dot}background:{_REPLICA_COLOUR}"></span>Replica (●)</span>'
                    f'<span><span style="{_legend_dot}background:{_BYZANTINE_COLOUR}"></span>Byzantine (✕)</span>'
                    f'<span><span style="{_legend_dot}background:{_CANDIDATE_COLOUR}"></span>Candidate (◆)</span>'
                    f'</div>'
                )

            # RIGHT — metrics + buttons + log (fixed height, no expanding content here)
            with gr.Column(scale=2, min_width=280):

                metrics_html = gr.HTML(value=_metrics_html(None, None))

                gr.HTML(f'<div style="{_section_title_style};margin-top:14px">Operations</div>')
                with gr.Row():
                    btn_reset = gr.Button("⟳  Reset",     elem_classes="btn-reset", size="sm")
                    btn_step  = gr.Button("▶  Step",      elem_classes="btn-step",  size="sm")
                    btn_state = gr.Button("◎  Get State", elem_classes="btn-state", size="sm")

                gr.HTML(f'<div style="{_section_title_style};margin-top:14px">Activity Log</div>')
                log_box = gr.Textbox(
                    value="Ready — click Reset to begin.",
                    label="",
                    show_label=False,
                    lines=7,
                    max_lines=7,
                    interactive=False,
                    container=False,
                )

        # ── BOTTOM ROW: full-width JSON (expanding this never touches the graph) ──
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Full State JSON", open=False):
                    state_json_box = gr.Code(
                        value="null",
                        language="json",
                        label="",
                        show_label=False,
                        interactive=False,
                    )

        # Pin the left column to the top so it doesn't stretch when right grows
        gr.HTML(
            "<style>"
            "#accordis-left-col { align-self: flex-start !important; }"
            "</style>"
        )

        # ── Wire buttons ───────────────────────────────────────────────────
        _outputs = [cluster_plot, metrics_html, log_box, state_json_box]

        btn_reset.click(fn=_do_reset,    inputs=[], outputs=_outputs)
        btn_step.click( fn=_do_step,     inputs=[], outputs=_outputs)
        btn_state.click(fn=_do_get_state,inputs=[], outputs=_outputs)

    return demo
