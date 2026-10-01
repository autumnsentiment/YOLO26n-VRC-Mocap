#!/usr/bin/env python3
"""Desktop entry point for the YOLO26n VRC Mocap dashboard."""

import argparse
import os
import sys
import webbrowser
from pathlib import Path


def _prepare_paths() -> Path:
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
        bin_dir = root / "bin"
    else:
        bin_dir = Path(__file__).resolve().parent
        root = bin_dir.parent

    if str(bin_dir) not in sys.path:
        sys.path.insert(0, str(bin_dir))
    os.chdir(root)
    return root


def _open_browser_later(url: str, delay: float):
    def _open():
        import time

        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    import threading

    threading.Thread(target=_open, daemon=True).start()


def main():
    _prepare_paths()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True, write_through=True)
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="YOLO26n VRC Mocap Dashboard Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8766, help="Dashboard port")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    parser.add_argument("--open-browser", action="store_true", help="Open the system browser after startup")
    parser.add_argument("--mocap-child", action="store_true", help=argparse.SUPPRESS)
    args, child_args = parser.parse_known_args()

    if args.mocap_child:
        from main import main as mocap_main
        sys.argv = [sys.argv[0], *child_args]
        mocap_main()
        return

    from dashboard_server import run_server

    url = f"http://{args.host}:{args.port}"
    print(f"YOLO26n VRC Mocap Dashboard: {url}")
    auto_open = args.open_browser or os.environ.get("YOLO26N_OPEN_BROWSER") == "1"
    if auto_open and not args.no_browser:
        _open_browser_later(url, 1.0)
    run_server(args.host, args.port)


if __name__ == "__main__":
    main()
