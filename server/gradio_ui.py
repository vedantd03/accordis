"""Custom Gradio UI for the Accordis BFT consensus environment.

Builds a "Visualizer" tab shown alongside the default OpenEnv Playground tab.

Layout
──────
Header bar  — episode status, step counter, live health badge
Left  col   — Plotly cluster topology graph + legend
Right col   — Live metrics chips  /  Byzantine activity panel  /  buttons  /  log
Bottom row  — Honest-node table  |  Full state JSON (accordions)
"""

from __future__ import annotations

import json
import math
import asyncio
from typing import Any, Dict, List, Optional

import gradio as gr
from accordis.server.tasks.task_easy import EasyTask
from accordis.server.tasks.task_medium import MediumTask
from accordis.server.tasks.task_hard import HardTask

try:
    import plotly.graph_objects as go
    _PLOTLY_OK = True
except ImportError:
    _PLOTLY_OK = False


# ── Colour palette ─────────────────────────────────────────────────────────
_DARK_BG        = "#0d1117"
_CARD_BG        = "#161b22"
_CARD_BG2       = "#1c2128"
_BORDER         = "#30363d"
_BORDER_LIGHT   = "#21262d"
_ACCENT_BLUE    = "#58a6ff"
_ACCENT_GREEN   = "#3fb950"
_ACCENT_RED     = "#f85149"
_ACCENT_AMBER   = "#d29922"
_ACCENT_PURPLE  = "#bc8cff"
_ACCENT_CYAN    = "#39c5cf"
_TEXT_PRIMARY   = "#e6edf3"
_TEXT_SECONDARY = "#8b949e"
_TEXT_MUTED     = "#484f58"

_LEADER_COLOUR   = _ACCENT_BLUE
_REPLICA_COLOUR  = _ACCENT_GREEN
_BYZANTINE_COLOUR = _ACCENT_RED
_CANDIDATE_COLOUR = _ACCENT_AMBER

_CSS = f"""
body, .gradio-container {{
    background: {_DARK_BG} !important;
    color: {_TEXT_PRIMARY} !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}}
.gr-box, .gr-panel {{ background: {_CARD_BG} !important; border: 1px solid {_BORDER} !important; }}
.btn-reset  {{ background: {_ACCENT_PURPLE} !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }}
.btn-step   {{ background: {_ACCENT_BLUE}   !important; color: #fff !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }}
.btn-state  {{ background: {_CARD_BG2}      !important; color: {_TEXT_PRIMARY} !important; border: 1px solid {_BORDER} !important; border-radius: 8px !important; }}
#accordis-left-col {{ align-self: flex-start !important; }}
"""

# ── Shared inline style helpers ─────────────────────────────────────────────

def _s(**kv: str) -> str:
    return ";".join(f"{k.replace('_', '-')}:{v}" for k, v in kv.items())


_SECTION_TITLE = _s(
    font_size="10px", font_weight="700", letter_spacing="0.1em",
    text_transform="uppercase", color=_TEXT_SECONDARY, margin_bottom="10px",
    display="block",
)
_CHIP_BASE = _s(
    display="inline-flex", flex_direction="column", gap="2px",
    background=_DARK_BG, border=f"1px solid {_BORDER}",
    border_radius="8px", padding="8px 14px", min_width="72px",
)
_LABEL_STYLE = _s(
    font_size="9px", font_weight="700", letter_spacing="0.08em",
    text_transform="uppercase", color=_TEXT_SECONDARY,
)
_ROW_STYLE = _s(display="flex", flex_wrap="wrap", gap="8px", margin_bottom="14px")


# ── Graph ──────────────────────────────────────────────────────────────────

def _ring_positions(n: int) -> List[tuple[float, float]]:
    return [
        (math.cos(2 * math.pi * i / n - math.pi / 2),
         math.sin(2 * math.pi * i / n - math.pi / 2))
        for i in range(n)
    ]


def _build_node_graph(state: Optional[Dict[str, Any]]) -> "go.Figure":
    if not _PLOTLY_OK:
        fig = go.Figure()
        fig.add_annotation(text="plotly not installed", x=0.5, y=0.5, showarrow=False)
        return fig

    node_states: Dict[str, Any] = {}
    bfa_strategy = "none"
    step_num = 0
    curriculum = 1

    if state:
        node_states  = state.get("node_states", {})
        bfa_strategy = str(state.get("bfa_strategy", "none")).replace("BFAStrategy.", "")
        step_num     = state.get("step", 0)
        curriculum   = state.get("curriculum_level", 1)

    if not node_states:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=_CARD_BG, plot_bgcolor=_CARD_BG,
            font=dict(color=_TEXT_MUTED),
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            annotations=[dict(
                text="Click <b>Reset</b> to initialise the cluster",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=15, color=_TEXT_SECONDARY),
            )],
        )
        return fig

    node_ids  = list(node_states.keys())
    positions = _ring_positions(len(node_ids))

    # Edges
    honest_ex, honest_ey, byz_ex, byz_ey = [], [], [], []
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            ns_i = node_states[node_ids[i]]
            ns_j = node_states[node_ids[j]]
            x0, y0 = positions[i]
            x1, y1 = positions[j]
            if ns_i.get("is_byzantine") or ns_j.get("is_byzantine"):
                byz_ex += [x0, x1, None]; byz_ey += [y0, y1, None]
            else:
                honest_ex += [x0, x1, None]; honest_ey += [y0, y1, None]

    traces = []
    if honest_ex:
        traces.append(go.Scatter(
            x=honest_ex, y=honest_ey, mode="lines",
            line=dict(width=1.4, color="rgba(88,166,255,0.15)"),
            hoverinfo="none", showlegend=False,
        ))
    if byz_ex:
        traces.append(go.Scatter(
            x=byz_ex, y=byz_ey, mode="lines",
            line=dict(width=1, color="rgba(248,81,73,0.18)", dash="dot"),
            hoverinfo="none", showlegend=False,
        ))

    node_x, node_y, node_colours, node_sizes = [], [], [], []
    node_symbols: List[str] = []
    hover_texts: List[str] = []
    node_labels: List[str] = []
    border_colours: List[str] = []

    for idx, nid in enumerate(node_ids):
        ns   = node_states[nid]
        x, y = positions[idx]
        node_x.append(x); node_y.append(y)
        node_labels.append(nid)

        is_byz = ns.get("is_byzantine", False)
        role   = ns.get("current_role", "replica")
        view   = ns.get("current_view", 0)
        cfg    = ns.get("config", {})
        log_len = len(ns.get("committed_log", []))

        if is_byz:
            colour, border, symbol, size = _BYZANTINE_COLOUR, "#7f2020", "x", 24
        elif role == "leader":
            colour, border, symbol, size = _LEADER_COLOUR, "#1e3a6e", "star", 32
        elif role == "candidate":
            colour, border, symbol, size = _CANDIDATE_COLOUR, "#5a3e0a", "diamond", 26
        else:
            colour, border, symbol, size = _REPLICA_COLOUR, "#0d3320", "circle", 22

        node_colours.append(colour)
        border_colours.append(border)
        node_symbols.append(symbol)
        node_sizes.append(size)

        vto  = cfg.get("view_timeout_ms",             "—")
        pd_  = cfg.get("pipeline_depth",              "—")
        rbs  = cfg.get("replication_batch_size",      "—")
        eq_t = cfg.get("equivocation_threshold",      "—")
        vat  = cfg.get("vote_aggregation_timeout_ms", "—")
        role_label = "BYZANTINE" if is_byz else role.upper()
        hover_texts.append(
            f"<b>{nid}</b><br>"
            f"Role: <b>{role_label}</b><br>"
            f"View: {view}  ·  Committed: {log_len} blocks<br>"
            f"──────────────────────<br>"
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
            color=node_colours, size=node_sizes, symbol=node_symbols,
            line=dict(width=2, color=border_colours),
        ),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=11, color=_TEXT_PRIMARY),
        hovertext=hover_texts, hoverinfo="text",
        showlegend=False,
    ))

    bfa_clean = bfa_strategy.replace("BFAStrategy.", "")
    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor=_CARD_BG, plot_bgcolor=_CARD_BG,
        font=dict(color=_TEXT_PRIMARY, family="Inter, sans-serif"),
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.5, 1.5], scaleanchor="x"),
        margin=dict(l=10, r=10, t=36, b=10),
        hoverlabel=dict(bgcolor=_DARK_BG, bordercolor=_BORDER,
                        font=dict(color=_TEXT_PRIMARY, size=12)),
        title=dict(
            text=(
                f"<b>Cluster</b>  ·  Step {step_num}  ·  "
                f"Level {curriculum}  ·  BFA: <i>{bfa_clean}</i>"
            ),
            font=dict(size=13, color=_TEXT_SECONDARY),
            x=0.5, xanchor="center",
        ),
    )
    return fig


# ── Metrics panel ──────────────────────────────────────────────────────────

def _chip(label: str, value: str, colour: str = _TEXT_PRIMARY, bg: str = _DARK_BG) -> str:
    base = _s(
        display="inline-flex", flex_direction="column", gap="2px",
        background=bg, border=f"1px solid {_BORDER}",
        border_radius="8px", padding="8px 14px", min_width="72px",
    )
    return (
        f'<div style="{base}">'
        f'<span style="{_LABEL_STYLE}">{label}</span>'
        f'<span style="font-size:17px;font-weight:700;color:{colour};line-height:1.2">{value}</span>'
        f'</div>'
    )


def _metrics_html(state: Optional[Dict[str, Any]], obs: Optional[Dict[str, Any]]) -> str:
    if not state:
        return (
            f'<span style="{_SECTION_TITLE}">Live Metrics</span>'
            f'<p style="color:{_TEXT_SECONDARY};font-size:13px;margin:0">No data — click Reset first.</p>'
        )

    step_num    = state.get("step", 0)
    curriculum  = state.get("curriculum_level", 1)
    n_nodes     = state.get("n_nodes", 0)
    f_byzantine = state.get("f_byzantine", 0)
    bfa         = str(state.get("bfa_strategy", "none")).replace("BFAStrategy.", "")
    vc          = state.get("view_change_count", 0)
    fin_txns    = state.get("finalized_txn_count", 0)

    # Throughput from obs
    avg_tps = 0.0
    nodes_obs: Dict[str, Any] = {}
    if obs and isinstance(obs.get("nodes"), dict):
        nodes_obs = obs["nodes"]
    elif obs and isinstance(obs.get("observation"), dict):
        nodes_obs = obs["observation"].get("nodes", {})
    if nodes_obs:
        tps_vals = [n.get("commit_throughput_tps", 0.0) for n in nodes_obs.values()]
        avg_tps = sum(tps_vals) / max(len(tps_vals), 1)

    bfa_colour = _ACCENT_RED if bfa not in ("none", "NONE") else _ACCENT_GREEN
    byz_colour = _ACCENT_RED if f_byzantine else _ACCENT_GREEN
    vc_colour  = _ACCENT_AMBER if vc else _TEXT_PRIMARY

    chips = "".join([
        _chip("Step",           str(step_num)),
        _chip("Curriculum",     str(curriculum), _ACCENT_CYAN),
        _chip("Nodes",          str(n_nodes)),
        _chip("Byzantine",      str(f_byzantine), byz_colour),
        _chip("BFA Strategy",   bfa,              bfa_colour),
        _chip("View Changes",   str(vc),          vc_colour),
        _chip("Finalized Txns", str(fin_txns),    _ACCENT_BLUE),
        _chip("Avg TPS",        f"{avg_tps:.2f}", _ACCENT_GREEN),
    ])

    return (
        f'<span style="{_SECTION_TITLE}">Live Metrics</span>'
        f'<div style="{_ROW_STYLE}">{chips}</div>'
    )


# ── Byzantine activity panel ───────────────────────────────────────────────

def _byzantine_panel_html(
    state: Optional[Dict[str, Any]],
    obs: Optional[Dict[str, Any]],
) -> str:
    """
    Shows Byzantine node visibility from the full AccordisState.

    Why state, not obs?
    ─────────────────────────────────────────────────────────────────────
    Observations (reset/step return value) are *agent-scoped*: only honest
    nodes appear there.  Byzantine nodes are deliberately excluded from the
    observation space — partial observability is the whole point.

    The AccordisState (server-side ground truth) stores a NodeState entry
    for every node, including Byzantine ones.  That is what we surface here
    so the *dashboard operator* can see what the adversary is doing, while
    the *agent* still sees only its partial view.
    """
    title = f'<span style="{_SECTION_TITLE}">Byzantine Activity</span>'

    if not state:
        return (
            title +
            f'<p style="color:{_TEXT_SECONDARY};font-size:13px;margin:0">'
            f'No data — click Reset first.</p>'
        )

    node_states: Dict[str, Any] = state.get("node_states", {})
    bfa_strategy = str(state.get("bfa_strategy", "none")).replace("BFAStrategy.", "")
    f_byzantine  = state.get("f_byzantine", 0)

    byz_nodes = {nid: ns for nid, ns in node_states.items() if ns.get("is_byzantine")}

    # Build per-honest-node suspicion map from obs
    suspicion_counts: Dict[str, int] = {}  # byz_nid → how many honest nodes suspect it
    nodes_obs: Dict[str, Any] = {}
    if obs and isinstance(obs.get("nodes"), dict):
        nodes_obs = obs["nodes"]
    elif obs and isinstance(obs.get("observation"), dict):
        nodes_obs = obs["observation"].get("nodes", {})

    for hon_obs in nodes_obs.values():
        for peer_id, suspected in hon_obs.get("suspected_byzantine", {}).items():
            if suspected:
                suspicion_counts[peer_id] = suspicion_counts.get(peer_id, 0) + 1

    if f_byzantine == 0 or not byz_nodes:
        bfa_col = _ACCENT_GREEN
        badge = (
            f'<span style="background:{_ACCENT_GREEN}22;color:{_ACCENT_GREEN};'
            f'border:1px solid {_ACCENT_GREEN}44;border-radius:5px;'
            f'padding:2px 8px;font-size:11px;font-weight:600">No adversary active</span>'
        )
        return title + badge

    # BFA strategy badge
    bfa_col = _ACCENT_RED if bfa_strategy not in ("none", "NONE") else _ACCENT_GREEN
    strategy_badge = (
        f'<span style="background:{bfa_col}22;color:{bfa_col};'
        f'border:1px solid {bfa_col}44;border-radius:5px;'
        f'padding:2px 8px;font-size:11px;font-weight:600;margin-bottom:10px;display:inline-block">'
        f'Strategy: {bfa_strategy}</span>'
    )

    # Per-byzantine-node rows
    _row_base = _s(
        display="flex", align_items="center", justify_content="space_between",
        gap="12px", background=_DARK_BG, border=f"1px solid {_BORDER}",
        border_radius="8px", padding="8px 12px", margin_bottom="6px",
        font_size="12px",
    )
    _tag_base = _s(
        border_radius="4px", padding="2px 7px", font_size="10px",
        font_weight="600", white_space="nowrap",
    )

    rows = ""
    for nid, ns in byz_nodes.items():
        view     = ns.get("current_view", 0)
        log_len  = len(ns.get("committed_log", []))
        suspects = suspicion_counts.get(nid, 0)
        total_hon = len(nodes_obs) if nodes_obs else (state.get("n_nodes", 0) - f_byzantine)

        sus_col = _ACCENT_AMBER if suspects > 0 else _TEXT_SECONDARY
        sus_label = f"{suspects}/{total_hon} honest nodes suspect" if suspects else "Not yet suspected"

        rows += (
            f'<div style="{_row_base}">'
            f'  <span style="color:{_ACCENT_RED};font-weight:700;font-family:monospace">{nid}</span>'
            f'  <span style="color:{_TEXT_SECONDARY}">view <b style="color:{_TEXT_PRIMARY}">{view}</b></span>'
            f'  <span style="color:{_TEXT_SECONDARY}">log <b style="color:{_TEXT_PRIMARY}">{log_len}</b></span>'
            f'  <span style="{_tag_base}background:{sus_col}22;color:{sus_col};border:1px solid {sus_col}44">'
            f'    {sus_label}</span>'
            f'</div>'
        )

    return title + strategy_badge + f'<div style="margin-top:8px">{rows}</div>'


# ── Honest-node table ──────────────────────────────────────────────────────

def _honest_table_html(obs: Optional[Dict[str, Any]]) -> str:
    title = f'<span style="{_SECTION_TITLE}">Honest Node Observations</span>'

    nodes_obs: Dict[str, Any] = {}
    if obs and isinstance(obs.get("nodes"), dict):
        nodes_obs = obs["nodes"]
    elif obs and isinstance(obs.get("observation"), dict):
        nodes_obs = obs["observation"].get("nodes", {})

    if not nodes_obs:
        return (
            title +
            f'<p style="color:{_TEXT_SECONDARY};font-size:13px;margin:0">'
            f'No observation data yet.</p>'
        )

    _th = _s(
        padding="6px 10px", text_align="left", font_size="9px",
        font_weight="700", letter_spacing="0.08em", text_transform="uppercase",
        color=_TEXT_SECONDARY, border_bottom=f"1px solid {_BORDER}",
        white_space="nowrap",
    )
    _td = _s(
        padding="6px 10px", font_size="12px",
        border_bottom=f"1px solid {_BORDER_LIGHT}", white_space="nowrap",
    )
    _table = _s(
        width="100%", border_collapse="collapse",
        background=_DARK_BG, border_radius="8px", overflow="hidden",
    )

    headers = ["Node", "Role", "View", "TPS", "Pipeline util", "QC misses", "View Δ (recent)", "Pending txns"]
    thead = "".join(f'<th style="{_th}">{h}</th>' for h in headers)

    rows = ""
    for nid, n in nodes_obs.items():
        role = n.get("current_role", "?")
        view = n.get("current_view", "?")
        tps  = n.get("commit_throughput_tps", 0.0)
        pu   = n.get("pipeline_utilisation", 0.0)
        qcm  = n.get("qc_formation_miss_streak", 0)
        vc_r = n.get("view_change_count_recent", 0)
        pend = n.get("pending_txn_count", 0)

        role_col = _ACCENT_BLUE if role == "leader" else (_ACCENT_AMBER if role == "candidate" else _ACCENT_GREEN)
        tps_col  = _ACCENT_GREEN if tps > 0.5 else (_ACCENT_AMBER if tps > 0.1 else _TEXT_SECONDARY)
        vc_col   = _ACCENT_AMBER if vc_r > 0 else _TEXT_PRIMARY

        rows += (
            f'<tr>'
            f'<td style="{_td}font-family:monospace;color:{_TEXT_PRIMARY}">{nid}</td>'
            f'<td style="{_td}"><span style="color:{role_col};font-weight:600;text-transform:uppercase;font-size:10px">{role}</span></td>'
            f'<td style="{_td}color:{_TEXT_SECONDARY}">{view}</td>'
            f'<td style="{_td}color:{tps_col};font-weight:600">{tps:.2f}</td>'
            f'<td style="{_td}color:{_TEXT_SECONDARY}">{pu:.1%}</td>'
            f'<td style="{_td}color:{"#f85149" if qcm > 2 else _TEXT_PRIMARY}">{qcm}</td>'
            f'<td style="{_td}color:{vc_col}">{vc_r}</td>'
            f'<td style="{_td}color:{_TEXT_SECONDARY}">{pend}</td>'
            f'</tr>'
        )

    table = (
        f'<table style="{_table}">'
        f'<thead><tr>{thead}</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table>'
    )
    return title + table


# ── Log formatter ──────────────────────────────────────────────────────────

_ACTION_HEADERS = [
    "Node",
    "view_timeout_ms",
    "pipeline_depth",
    "replication_batch_size",
    "equivocation_threshold",
    "vote_aggregation_timeout_ms",
]


def _action_rows_from_state(state: Optional[Dict[str, Any]]) -> List[List[Any]]:
    if not state:
        return []

    node_states: Dict[str, Any] = state.get("node_states", {})
    rows: List[List[Any]] = []
    for nid, ns in node_states.items():
        if ns.get("is_byzantine"):
            continue
        cfg = ns.get("config", {})
        rows.append([
            nid,
            cfg.get("view_timeout_ms", 1000),
            cfg.get("pipeline_depth", 2),
            cfg.get("replication_batch_size", 64),
            cfg.get("equivocation_threshold", 5),
            cfg.get("vote_aggregation_timeout_ms", 500),
        ])

    rows.sort(key=lambda row: row[0])
    return rows


def _coerce_int(value: Any, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _action_payload_from_rows(rows: Any) -> Dict[str, Any]:
    nodes_action: Dict[str, Any] = {}
    if not isinstance(rows, list):
        return {"nodes": nodes_action}

    for row in rows:
        if not isinstance(row, list) or len(row) < len(_ACTION_HEADERS):
            continue

        nid = str(row[0]).strip()
        if not nid:
            continue

        nodes_action[nid] = {
            "node_id": nid,
            "view_timeout_ms": _coerce_int(row[1], 1000),
            "pipeline_depth": _coerce_int(row[2], 2),
            "replication_batch_size": _coerce_int(row[3], 64),
            "equivocation_threshold": _coerce_int(row[4], 5),
            "vote_aggregation_timeout_ms": _coerce_int(row[5], 500),
        }

    return {"nodes": nodes_action}


_TASK_BUILDERS = {
    "easy": EasyTask,
    "medium": MediumTask,
    "hard": HardTask,
}


def _task_reset_payload(task_name: str) -> Dict[str, Any]:
    builder = _TASK_BUILDERS.get(str(task_name).lower())
    if builder is None:
        return {}
    return builder().get_initial_conditions()


def _task_preset_summary(payload: Optional[Dict[str, Any]], task_name: str = "") -> str:
    title = f'<span style="{_SECTION_TITLE}">Preset Details</span>'
    if not payload:
        return (
            title +
            f'<p style="color:{_TEXT_SECONDARY};font-size:12px;margin:0">'
            "Choose a task preset to load its reset configuration.</p>"
        )

    task_label = str(task_name or payload.get("task", "")).upper()
    chips = "".join([
        _chip("Task", task_label, _ACCENT_CYAN),
        _chip("Curriculum", str(payload.get("curriculum_level", "—")), _ACCENT_BLUE),
        _chip("Leader Rotation", str(payload.get("leader_rotation", "—")).replace("LeaderRotation.", ""), _ACCENT_GREEN),
        _chip("Max Steps", str(payload.get("max_steps", "—")), _ACCENT_AMBER),
    ])
    return title + f'<div style="{_ROW_STYLE}">{chips}</div>'


def _preview_state_from_reset_payload(payload: Optional[Dict[str, Any]], task_name: str = "") -> Optional[Dict[str, Any]]:
    if not payload:
        return None

    n_nodes = max(1, int(payload.get("n_nodes", 4)))
    f_byzantine = max(0, min(int(payload.get("f_byzantine", 0)), n_nodes - 1))
    curriculum = int(payload.get("curriculum_level", 1))

    node_states: Dict[str, Any] = {}
    for idx in range(n_nodes):
        nid = f"node_{idx}"
        is_byzantine = idx >= (n_nodes - f_byzantine)
        role = "leader" if idx == 0 else "replica"
        node_states[nid] = {
            "node_id": nid,
            "is_byzantine": is_byzantine,
            "current_view": 0,
            "current_role": role,
            "committed_log": [],
            "config": {
                "view_timeout_ms": 1000,
                "pipeline_depth": 2,
                "replication_batch_size": 64,
                "equivocation_threshold": 5,
                "vote_aggregation_timeout_ms": 500,
            },
        }

    return {
        "episode_id": f"preset-{task_name or 'preview'}",
        "step": 0,
        "curriculum_level": curriculum,
        "n_nodes": n_nodes,
        "f_byzantine": f_byzantine,
        "bfa_strategy": "none",
        "view_change_count": 0,
        "node_states": node_states,
    }


def _fmt_log(operation: str, data: Any) -> str:
    if isinstance(data, str):
        return f"[{operation.upper()}] ERROR — {data}"
    if not data:
        return f"[{operation.upper()}] (no data)"

    lines = [f"[{operation.upper()}]"]

    if operation == "reset":
        obs   = data.get("observation", data)
        nodes = obs.get("nodes", {}) if isinstance(obs, dict) else {}
        lines.append(f"  nodes: {list(nodes.keys()) or '—'}")
        lines.append(f"  done={data.get('done')}  reward={data.get('reward')}")

    elif operation == "step":
        obs   = data.get("observation", data)
        nodes = obs.get("nodes", {}) if isinstance(obs, dict) else {}
        for nid, nobs in list(nodes.items())[:8]:
            role = nobs.get("current_role", "?")
            view = nobs.get("current_view", "?")
            tps  = nobs.get("commit_throughput_tps", 0.0)
            lines.append(f"  {nid}: role={role}  view={view}  tps={tps:.2f}")
        if len(nodes) > 8:
            lines.append(f"  … (+{len(nodes)-8} more)")
        lines.append(f"  done={data.get('done')}  reward={data.get('reward')}")
        meta = data.get("metadata") or data.get("info") or {}
        if meta:
            bfa  = meta.get("bfa_strategy", "")
            lv   = meta.get("liveness_rate", "")
            lines.append(f"  bfa={bfa}  liveness_rate={lv}")

    elif operation == "get_state":
        lines.append(f"  step={data.get('step','?')}  "
                     f"bfa={data.get('bfa_strategy','?')}  "
                     f"fin_txns={data.get('finalized_txn_count','?')}")

    return "\n".join(lines)


# ── Header bar ─────────────────────────────────────────────────────────────

def _header_html(state: Optional[Dict[str, Any]]) -> str:
    if not state:
        health_badge = (
            f'<span style="background:{_ACCENT_AMBER}22;color:{_ACCENT_AMBER};'
            f'border:1px solid {_ACCENT_AMBER}55;border-radius:6px;'
            f'padding:3px 10px;font-size:11px;font-weight:600">IDLE</span>'
        )
        episode = "—"
        step_info = ""
    else:
        bfa   = str(state.get("bfa_strategy", "none")).replace("BFAStrategy.", "")
        step  = state.get("step", 0)
        eid   = state.get("episode_id", "")[:8]
        n     = state.get("n_nodes", 0)
        f_byz = state.get("f_byzantine", 0)
        vc    = state.get("view_change_count", 0)

        if bfa not in ("none", "NONE") and f_byz:
            health_col = _ACCENT_RED
            health_lbl = "UNDER ATTACK"
        elif vc > 3:
            health_col = _ACCENT_AMBER
            health_lbl = "UNSTABLE"
        else:
            health_col = _ACCENT_GREEN
            health_lbl = "HEALTHY"

        health_badge = (
            f'<span style="background:{health_col}22;color:{health_col};'
            f'border:1px solid {health_col}55;border-radius:6px;'
            f'padding:3px 10px;font-size:11px;font-weight:600">{health_lbl}</span>'
        )
        episode  = eid
        step_info = (
            f'<span style="font-size:12px;color:{_TEXT_SECONDARY};margin-left:16px">'
            f'step <b style="color:{_TEXT_PRIMARY}">{step}</b> · '
            f'{n} nodes · {f_byz} Byzantine</span>'
        )

    return (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'padding:14px 0 12px;border-bottom:1px solid {_BORDER};margin-bottom:16px">'
        f'  <div>'
        f'    <span style="font-size:22px;font-weight:800;color:{_TEXT_PRIMARY};letter-spacing:-0.5px">Accordis</span>'
        f'    <span style="font-size:12px;color:{_TEXT_SECONDARY};margin-left:10px">BFT Consensus Dashboard</span>'
        f'    {step_info}'
        f'  </div>'
        f'  <div style="display:flex;align-items:center;gap:12px">'
        f'    {health_badge}'
        f'    <span style="font-size:11px;color:{_TEXT_MUTED};font-family:monospace">ep: {episode}</span>'
        f'  </div>'
        f'</div>'
    )


# ── Main builder ───────────────────────────────────────────────────────────

def build_accordis_gradio_app(
    web_manager: Any,
    action_fields: List[Dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """Returns a gr.Blocks that becomes the 'Custom' tab in the TabbedInterface."""

    _shared: Dict[str, Any] = {
        "state":    None,
        "last_obs": None,
        "log":      "Ready — click Reset to begin.",
        "action_rows": [],
        "task_name": "",
        "reset_payload": {},
    }

    # ── Callbacks ──────────────────────────────────────────────────────────

    def _load_task_preset(task_name: str):
        payload = _task_reset_payload(task_name)
        _shared["task_name"] = str(task_name).lower() if payload else ""
        _shared["reset_payload"] = payload
        preview_state = _preview_state_from_reset_payload(payload, _shared["task_name"])
        if not payload:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                _task_preset_summary(None),
                _header_html(None),
                _build_node_graph(None),
            )

        return (
            payload.get("n_nodes", 4),
            payload.get("f_byzantine", 0),
            payload.get("pool_size", 1000),
            _task_preset_summary(payload, _shared["task_name"]),
            _header_html(preview_state),
            _build_node_graph(preview_state),
        )

    async def _do_reset(task_name: str, n_nodes: float, f_byzantine: float, pool_size: float):
        n_nodes_i = max(1, int(n_nodes or 4))
        f_byzantine_i = max(0, min(int(f_byzantine or 0), n_nodes_i - 1))
        pool_size_i = max(1, int(pool_size or 1000))

        try:
            preset_payload = _task_reset_payload(task_name)
            _shared["task_name"] = str(task_name).lower() if preset_payload else ""
            _shared["reset_payload"] = preset_payload
            reset_environment_payload = {
                **preset_payload,
                "n_nodes": n_nodes_i,
                "f_byzantine": f_byzantine_i,
                "pool_size": pool_size_i,
            }
            data = await web_manager.reset_environment(reset_environment_payload)
        except Exception as exc:
            data = str(exc)
        try:
            state = web_manager.get_state()
        except Exception:
            state = None
        _shared["state"]    = state
        _shared["last_obs"] = data if isinstance(data, dict) else None
        _shared["log"]      = _fmt_log("reset", data)
        _shared["action_rows"] = _action_rows_from_state(state)
        return _refresh_ui()

    async def _do_step(action_rows: Any):
        if _shared["state"] is None:
            _shared["log"] = "[STEP] Not started — click Reset first."
            return _refresh_ui()
        _shared["action_rows"] = action_rows if isinstance(action_rows, list) else _shared["action_rows"]
        try:
            data = await web_manager.step_environment(_action_payload_from_rows(_shared["action_rows"]))
        except Exception as exc:
            data = str(exc)
        try:
            new_state = web_manager.get_state()
        except Exception:
            new_state = _shared["state"]
        _shared["state"]    = new_state
        _shared["last_obs"] = data if isinstance(data, dict) else None
        _shared["log"]      = _fmt_log("step", data)
        _shared["action_rows"] = _action_rows_from_state(new_state) or _shared["action_rows"]
        return _refresh_ui()

    def _do_get_state():
        try:
            state = web_manager.get_state()
        except Exception as exc:
            state = {"error": str(exc)}
        _shared["state"] = state
        _shared["log"]   = _fmt_log("get_state", state)
        _shared["action_rows"] = _action_rows_from_state(state) or _shared["action_rows"]
        return _refresh_ui()

    def _refresh_ui():
        state    = _shared["state"]
        last_obs = _shared["last_obs"]
        return (
            _header_html(state),
            _build_node_graph(state),
            _metrics_html(state, last_obs),
            _byzantine_panel_html(state, last_obs),
            _shared["log"],
            _honest_table_html(last_obs),
            json.dumps(state, indent=2, default=str) if state else "null",
            _shared["action_rows"],
            _task_preset_summary(_shared["reset_payload"], _shared["task_name"]),
        )

    # ── Layout ─────────────────────────────────────────────────────────────

    with gr.Blocks(title="Accordis — BFT Dashboard", css=_CSS) as demo:

        header_html = gr.HTML(value=_header_html(None))

        with gr.Row():

            # LEFT — topology
            with gr.Column(scale=3, min_width=380, elem_id="accordis-left-col"):
                gr.HTML(f'<span style="{_SECTION_TITLE}">Cluster Topology</span>')
                cluster_plot = gr.Plot(
                    value=_build_node_graph(None),
                    label="", show_label=False, container=False,
                )
                _dot = "display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:middle;"
                gr.HTML(
                    f'<div style="display:flex;gap:20px;flex-wrap:wrap;font-size:11px;'
                    f'color:{_TEXT_SECONDARY};margin-top:6px">'
                    f'<span><span style="{_dot}background:{_LEADER_COLOUR}"></span>Leader (★)</span>'
                    f'<span><span style="{_dot}background:{_REPLICA_COLOUR}"></span>Replica (●)</span>'
                    f'<span><span style="{_dot}background:{_BYZANTINE_COLOUR}"></span>Byzantine (✕)</span>'
                    f'<span><span style="{_dot}background:{_CANDIDATE_COLOUR}"></span>Candidate (◆)</span>'
                    f'</div>'
                )

            # RIGHT — controls & info
            with gr.Column(scale=2, min_width=300):
                metrics_html  = gr.HTML(value=_metrics_html(None, None))
                byzantine_html = gr.HTML(value=_byzantine_panel_html(None, None))

                gr.HTML(f'<span style="{_SECTION_TITLE};margin-top:14px">Controls</span>')
                with gr.Tabs():
                    with gr.Tab("Reset"):
                        gr.HTML(
                            f'<p style="color:{_TEXT_SECONDARY};font-size:12px;margin:0 0 12px">'
                            "Load an `easy`, `medium`, or `hard` task preset, then reset with those environment conditions."
                            "</p>"
                        )
                        task_preset = gr.Radio(
                            choices=["easy", "medium", "hard"],
                            value="easy",
                            label="Task preset",
                        )
                        btn_load_task = gr.Button("Load Task Preset", elem_classes="btn-state", size="sm")
                        with gr.Row():
                            inp_n_nodes = gr.Number(value=4, precision=0, label="n_nodes")
                            inp_f_byzantine = gr.Number(value=1, precision=0, label="f_byzantine")
                            inp_pool_size = gr.Number(value=1000, precision=0, label="pool_size")
                        preset_html = gr.HTML(value=_task_preset_summary(None))
                        btn_reset = gr.Button("⟳  Reset Episode", elem_classes="btn-reset", size="sm")

                    with gr.Tab("Action"):
                        gr.HTML(
                            f'<p style="color:{_TEXT_SECONDARY};font-size:12px;margin:0 0 12px">'
                            "Edit the honest-node config that will be sent on the next environment step. Byzantine nodes are excluded."
                            "</p>"
                        )
                        action_table = gr.Dataframe(
                            headers=_ACTION_HEADERS,
                            datatype=["str", "number", "number", "number", "number", "number"],
                            value=[],
                            type="array",
                            row_count=(0, "dynamic"),
                            col_count=(len(_ACTION_HEADERS), "fixed"),
                            interactive=True,
                            wrap=True,
                            label="Per-node action payload",
                        )
                        with gr.Row():
                            btn_step  = gr.Button("▶  Step With Action", elem_classes="btn-step",  size="sm")
                            btn_state = gr.Button("◎  Refresh State", elem_classes="btn-state", size="sm")

                gr.HTML(f'<span style="{_SECTION_TITLE};margin-top:14px">Activity Log</span>')
                log_box = gr.Textbox(
                    value="Ready — click Reset to begin.",
                    label="", show_label=False,
                    lines=7, max_lines=7,
                    interactive=False, container=False,
                )

        # BOTTOM — expanding panels
        with gr.Row():
            with gr.Column():
                with gr.Accordion("Honest Node Observations", open=True):
                    honest_table_html = gr.HTML(value=_honest_table_html(None))

        with gr.Row():
            with gr.Column():
                with gr.Accordion("Full State JSON", open=False):
                    state_json_box = gr.Code(
                        value="null", language="json",
                        label="", show_label=False, interactive=False,
                    )

        # Pin left column
        gr.HTML("<style>#accordis-left-col { align-self: flex-start !important; }</style>")

        # Wire buttons
        _outputs = [
            header_html, cluster_plot, metrics_html,
            byzantine_html, log_box, honest_table_html, state_json_box, action_table, preset_html,
        ]
        btn_load_task.click(
            fn=_load_task_preset,
            inputs=[task_preset],
            outputs=[inp_n_nodes, inp_f_byzantine, inp_pool_size, preset_html, header_html, cluster_plot],
        )
        task_preset.change(
            fn=_load_task_preset,
            inputs=[task_preset],
            outputs=[inp_n_nodes, inp_f_byzantine, inp_pool_size, preset_html, header_html, cluster_plot],
        )
        btn_reset.click(
            fn=_do_reset,
            inputs=[task_preset, inp_n_nodes, inp_f_byzantine, inp_pool_size],
            outputs=_outputs
        )
        btn_step.click(fn=_do_step, inputs=[action_table], outputs=_outputs)
        btn_state.click(fn=_do_get_state,inputs=[], outputs=_outputs)

    return demo
