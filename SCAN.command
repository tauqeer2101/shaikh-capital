#!/bin/bash
# Shaikh Capital — run the daily scan across US, Germany and India.
cd "$(dirname "$0")"
echo "Running the Shaikh Capital scan (US, Germany, India)..."
.venv/bin/python -m atlas2 scan
echo
echo "Done. Open the dashboard with RUN_DASHBOARD.command (or refresh it if already open)."
read -p "Press Enter to close..."
