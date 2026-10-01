#!/usr/bin/env python3
"""Non-destructive checks for runtime model import/listing API."""

import copy
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "bin"))

import dashboard_server


def main():
    original_config = copy.deepcopy(dashboard_server.load_config())
    app = dashboard_server.app
    client = app.test_client()

    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "codex_runtime_import_test.pt"
        source.write_bytes(b"runtime model api test")

        res = client.post(
            "/api/models/import",
            json={"source_path": str(source), "mode": "copy", "set_current": True},
        )
        assert res.status_code == 200, res.get_data(as_text=True)
        payload = res.get_json()
        assert payload["status"] == "ok", payload
        imported = Path(payload["model_path"])
        assert imported.exists(), imported
        assert imported.name == source.name

        res = client.get("/api/models")
        assert res.status_code == 200, res.get_data(as_text=True)
        models = res.get_json()["models"]
        assert any(m["name"] == source.name and m["current"] for m in models), models

        try:
            imported.unlink()
        except OSError:
            pass

    dashboard_server.save_config(original_config)
    print("Model API checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
