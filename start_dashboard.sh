#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

HOST="${YOLO26N_HOST:-0.0.0.0}"
PORT="${YOLO26N_PORT:-8765}"
EDITION="${YOLO26N_EDITION:-full}"

case "$EDITION" in
  full)
    exec uv run --frozen --extra full python bin/desktop_launcher.py --host "$HOST" --port "$PORT" --no-browser "$@"
    ;;
  lite)
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-lite}"
    exec uv run --frozen python bin/desktop_launcher.py --host "$HOST" --port "$PORT" --no-browser "$@"
    ;;
  *)
    echo "Unknown YOLO26N_EDITION=$EDITION (use full or lite)" >&2
    exit 1
    ;;
esac