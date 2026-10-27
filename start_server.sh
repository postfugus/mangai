#!/usr/bin/env bash
cd /root/mangakawaii-downloader
export PYTHONPATH=/root/mangakawaii-downloader
PORT=${1:-8080}
echo "[*] Starting Manga Web Server on port $PORT..."
exec uvicorn web_server:app --host 0.0.0.0 --port "$PORT"
