from openenv.core.env_server.serialization import serialize_observation

from accordis.server.accordis_environment import AccordisEnvironment
from accordis.server.gradio_ui import (
    _action_from_rows,
    _build_node_graph_html,
    _compact_state_for_display,
)


def test_action_from_rows_builds_multinode_payload():
    payload = _action_from_rows(
        [
            ["node_0", 2500, 3, 128, 7, 600],
            ["node_1", 2100, 2, 64, 5, 500],
        ]
    )

    assert sorted(payload["nodes"]) == ["node_0", "node_1"]
    assert payload["nodes"]["node_0"]["pipeline_depth"] == 3
    assert payload["nodes"]["node_0"]["replication_batch_size"] == 128
    assert payload["nodes"]["node_1"]["vote_aggregation_timeout_ms"] == 500


def test_compact_state_for_display_trims_large_state_fields():
    env = AccordisEnvironment()
    response = serialize_observation(env.reset(n_nodes=5, f_byzantine=1, pool_size=20))

    compact = _compact_state_for_display(env.state.model_dump(), response)

    assert compact["n_nodes"] == 5
    assert compact["honest_nodes"] == 4
    assert compact["byzantine_nodes"] == 1
    assert compact["txn_pool_size"] == 20
    assert compact["proposal_registry_size"] == 20
    assert "proposal_registry" not in compact


def test_build_node_graph_html_renders_nodes_and_hover_copy():
    env = AccordisEnvironment()
    response = serialize_observation(env.reset(n_nodes=4, f_byzantine=1, pool_size=12))

    graph_html = _build_node_graph_html(env.state.model_dump(), response["observation"])

    assert "Consensus Fabric" in graph_html
    assert "Hover nodes to inspect role" in graph_html
    assert "node_0" in graph_html
