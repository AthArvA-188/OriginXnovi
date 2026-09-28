"""The live agent swarm page embeds the swarm console, and the console serves from background threads."""

import json
import socket
import sys
import urllib.request
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "app" / "site_pages" / "agent_swarm.py"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_console_serves_from_background_threads():
    sys.path.insert(0, str(ROOT))
    from dashboard.server import start_background

    srv = start_background(0, interval=60, visual_backend="opencv")
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        with urllib.request.urlopen(base + "/api/state", timeout=10) as r:
            state = json.loads(r.read())
        assert {"agents", "concerns", "health", "tick"} <= state.keys()
        with urllib.request.urlopen(base + "/", timeout=10) as r:
            assert "Pavilion Cerebro" in r.read().decode("utf-8")
    finally:
        srv.shutdown()
        srv.server_close()


def test_agent_swarm_page_renders(monkeypatch):
    monkeypatch.setenv("CEREBRO_DASHBOARD_PORT", str(_free_port()))
    at = AppTest.from_file(str(PAGE), default_timeout=180).run()
    assert not at.exception
    assert not at.error, "the console did not start"
    assert any("Live agent swarm" in t.value for t in at.title)
    assert len(at.metric) == 4
