#!/usr/bin/env python3
"""Packaged desktop entry point that starts mocap directly without the Web UI."""

import os
import sys
from pathlib import Path


def _prepare_paths() -> Path:
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
        bin_path = root / "bin"
    else:
        bin_path = Path(__file__).resolve().parent
        root = bin_path.parent

    if str(bin_path) not in sys.path:
        sys.path.insert(0, str(bin_path))
    os.chdir(root)
    return root


def main():
    root = _prepare_paths()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass

    if "--config" not in sys.argv:
        runtime_config = root / "config" / "config.json"
        bundled_config = Path(getattr(sys, "_MEIPASS", root)) / "config" / "config.json"
        config_path = runtime_config if runtime_config.exists() else bundled_config
        if config_path.exists():
            sys.argv.extend(["--config", str(config_path)])

    from main import main as mocap_main

    mocap_main()


if __name__ == "__main__":
    main()
