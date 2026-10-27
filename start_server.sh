#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")"
export PYTHONPATH="$PWD"
export MANGA_DOWNLOAD_DIR="${MANGA_DOWNLOAD_DIR:-$PWD/downloads}"
PORT=${1:-8080}
echo "[*] Starting Manga Web Server on port $PORT..."
exec .venv/bin/uvicorn web_server:app --host 0.0.0.0 --port "$PORT"
