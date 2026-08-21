#!/bin/bash
# Shaikh Capital — open the dashboard.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "First run: creating Python environment..."
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet yfinance pandas numpy fastapi "uvicorn[standard]" lxml html5lib requests
fi
( sleep 2 && open "http://127.0.0.1:8899" ) &
echo "Shaikh Capital dashboard: http://127.0.0.1:8899  (leave this window open, Ctrl+C to stop)"
.venv/bin/python -m atlas2 serve
