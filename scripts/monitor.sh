#!/usr/bin/env bash
# One-shot monitor view for an in-flight CybOrg training run.
# Splits a tmux window into 4 panes:
#   - top-left:    nvidia-smi (refreshing every 2s)
#   - top-right:   tail -f train.log
#   - bottom-left: tail -f metrics.jsonl (one row per training step)
#   - bottom-right: htop
#
# Usage:
#   bash scripts/monitor.sh runs/red-red_easy-easy-YYYYMMDD-HHMMSS
#   bash scripts/monitor.sh                     # auto-picks latest runs/* dir
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(ls -dt runs/*/ 2>/dev/null | head -n1 || true)"
fi
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "No run directory found. Pass one as the first argument." >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not installed. Install with: sudo apt install -y tmux htop" >&2
  exit 1
fi

SESSION="cyborg-monitor"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n monitor

tmux send-keys -t "$SESSION":monitor.0 "watch -n 2 nvidia-smi" C-m
tmux split-window -h -t "$SESSION":monitor
tmux send-keys -t "$SESSION":monitor.1 "tail -f $RUN_DIR/train.log" C-m
tmux split-window -v -t "$SESSION":monitor.0
tmux send-keys -t "$SESSION":monitor.2 "tail -f $RUN_DIR/metrics.jsonl | sed -e 's/,/, /g'" C-m
tmux split-window -v -t "$SESSION":monitor.1
if command -v htop >/dev/null 2>&1; then
  tmux send-keys -t "$SESSION":monitor.3 "htop" C-m
else
  tmux send-keys -t "$SESSION":monitor.3 "top" C-m
fi

echo "Attached panes for: $RUN_DIR"
echo "Attach with:  tmux attach -t $SESSION"
echo "Detach with:  Ctrl-b then d"
tmux attach -t "$SESSION"
