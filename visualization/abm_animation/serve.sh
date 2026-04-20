#!/bin/bash
# Serve the ABM animation locally
# Open http://localhost:8080 in your browser
echo "Serving ABM animation at http://localhost:8080"
echo "Press Ctrl+C to stop"
cd "$(dirname "$0")"
python -m http.server 8080
