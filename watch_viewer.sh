#!/bin/bash
LOGDIR=~/microduck_rl/logs/rsl_rl/velocity/2026-09-30_19-08-35_velocity
LAST=""
VIEWER_PID=""

while true; do
  LATEST=$(ls -t "$LOGDIR"/model_*.pt 2>/dev/null | head -1)
  if [ -n "$LATEST" ] && [ "$LATEST" != "$LAST" ]; then
    if [ -n "$VIEWER_PID" ] && kill -0 "$VIEWER_PID" 2>/dev/null; then
      kill "$VIEWER_PID" 2>/dev/null
      wait "$VIEWER_PID" 2>/dev/null
      sleep 2
    fi
    cd ~/microduck_rl
    source "$HOME/.local/bin/env"
    uv run play Mjlab-Velocity-Flat-MicroDuck --checkpoint-file "$LATEST" --viewer viser > /tmp/viewer_watch.log 2>&1 &
    VIEWER_PID=$!
    LAST="$LATEST"
    echo "$(date '+%H:%M:%S'): switched viewer to $(basename "$LATEST") (pid $VIEWER_PID)"
  fi
  sleep 20
done
