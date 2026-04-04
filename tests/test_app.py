"""Tests for the HTTP and WebSocket endpoints exposed by openenv.

Endpoint behaviour summary:
  HTTP /reset  — stateless: creates fresh env, resets, returns observation. Always works.
  HTTP /step   — stateless: creates fresh env without reset → RuntimeError → 500.
  HTTP /state  — stateless: creates fresh env without reset → RuntimeError → 500.
  WS /ws       — stateful session: env persists across messages; reset → step → state work.
"""

import pytest
from starlette.testclient import TestClient

from accordis.server.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def client_no_raise():
    """Client that returns 500 responses instead of re-raising server exceptions."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _node_action(nid: str) -> dict:
    return {
        "node_id": nid,
        "view_timeout_ms": 2000,
        "pipeline_depth": 2,
        "replication_batch_size": 64,
        "equivocation_threshold": 5,
        "vote_aggregation_timeout_ms": 500,
    }


def _multi_node_action(node_ids: list) -> dict:
    return {"nodes": {nid: _node_action(nid) for nid in node_ids}}


# ── HTTP /reset ────────────────────────────────────────────────────────────────

class TestResetEndpoint:
    def test_reset_returns_200(self, client):
        resp = client.post("/reset")
        assert resp.status_code == 200

    def test_reset_response_has_required_fields(self, client):
        data = client.post("/reset").json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data

    def test_reset_observation_has_nodes(self, client):
        obs = client.post("/reset").json()["observation"]
        assert "nodes" in obs
        assert len(obs["nodes"]) > 0

    def test_reset_done_is_false(self, client):
        assert client.post("/reset").json()["done"] is False

    def test_reset_with_seed(self, client):
        resp = client.post("/reset", json={"seed": 42})
        assert resp.status_code == 200

    def test_reset_with_episode_id(self, client):
        resp = client.post("/reset", json={"episode_id": "test-episode-001"})
        assert resp.status_code == 200
        assert len(resp.json()["observation"]["nodes"]) > 0

    def test_reset_with_custom_node_count(self, client):
        resp = client.post("/reset", json={"n_nodes": 7, "f_byzantine": 2})
        assert resp.status_code == 200
        assert len(resp.json()["observation"]["nodes"]) == 5  # 7 - 2 honest


# ── HTTP /step ─────────────────────────────────────────────────────────────────

class TestStepEndpoint:
    def test_step_invalid_action_returns_422(self, client):
        resp = client.post("/step", json={"action": {"invalid": "data"}})
        assert resp.status_code == 422

    def test_step_without_prior_reset_returns_500(self, client_no_raise):
        # HTTP /step creates a fresh env per request — reset() is never called first.
        action = _multi_node_action(["node_0"])
        resp = client_no_raise.post("/step", json={"action": action})
        assert resp.status_code == 500


# ── HTTP /state ────────────────────────────────────────────────────────────────

class TestStateEndpoint:
    def test_state_without_prior_reset_returns_500(self, client):
        # HTTP /state creates a fresh env per request — state raises before reset().
        resp = client.get("/state")
        assert resp.status_code == 500


# ── WebSocket /ws ──────────────────────────────────────────────────────────────

class TestWebSocketEndpoints:
    def test_ws_reset_returns_observation(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {}})
            resp = ws.receive_json()
        assert resp["type"] == "observation"
        assert "nodes" in resp["data"]["observation"]
        assert len(resp["data"]["observation"]["nodes"]) > 0

    def test_ws_reset_done_is_false(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {}})
            resp = ws.receive_json()
        assert resp["data"]["done"] is False

    def test_ws_step_after_reset(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {}})
            reset_resp = ws.receive_json()
            node_ids = list(reset_resp["data"]["observation"]["nodes"].keys())

            ws.send_json({"type": "step", "data": _multi_node_action(node_ids)})
            step_resp = ws.receive_json()

        assert step_resp["type"] == "observation"
        assert "nodes" in step_resp["data"]["observation"]
        assert isinstance(step_resp["data"]["done"], bool)
        assert isinstance(step_resp["data"]["reward"], float)

    def test_ws_state_after_reset(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {}})
            ws.receive_json()

            ws.send_json({"type": "state"})
            state_resp = ws.receive_json()

        assert state_resp["type"] == "state"
        assert "episode_id" in state_resp["data"]
        assert state_resp["data"]["step"] == 0

    def test_ws_state_before_reset_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "state"})
            resp = ws.receive_json()
        assert resp["type"] == "error"

    def test_ws_step_before_reset_returns_error(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "step", "data": _multi_node_action(["node_0"])})
            resp = ws.receive_json()
        assert resp["type"] == "error"

    def test_ws_step_increments_step_count(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {}})
            reset_resp = ws.receive_json()
            node_ids = list(reset_resp["data"]["observation"]["nodes"].keys())
            action = _multi_node_action(node_ids)

            ws.send_json({"type": "step", "data": action})
            ws.receive_json()

            ws.send_json({"type": "state"})
            state_resp = ws.receive_json()

        assert state_resp["data"]["step"] == 1

    def test_ws_full_episode_reaches_done(self, client):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "reset", "data": {"max_steps": 3}})
            reset_resp = ws.receive_json()
            node_ids = list(reset_resp["data"]["observation"]["nodes"].keys())
            action = _multi_node_action(node_ids)

            done = False
            for _ in range(3):
                ws.send_json({"type": "step", "data": action})
                step_resp = ws.receive_json()
                done = step_resp["data"]["done"]
                if done:
                    break

        assert done is True
