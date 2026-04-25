#!/usr/bin/env bash
# Wrapper that runs CybOrg GRPO with the right defaults for a GPU droplet
# and tees stdout to runs/<task>/train.log so monitoring is trivial.
#
# Usage:
#   bash scripts/train.sh [task] [difficulty] [role] [episodes] [group_size] [model]
# Defaults:
#   task=red_easy difficulty=easy role=red episodes=200 group_size=4
#   model=Qwen/Qwen2.5-0.5B-Instruct
#
# The script is idempotent \u2014 each run gets its own output dir tagged by date.
set -euo pipefail

TASK="${1:-red_easy}"
DIFFICULTY="${2:-easy}"
ROLE="${3:-red}"
EPISODES="${4:-200}"
GROUP_SIZE="${5:-4}"
MODEL="${6:-Qwen/Qwen2.5-0.5B-Instruct}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/${ROLE}-${TASK}-${DIFFICULTY}-${STAMP}"
mkdir -p "$RUN_DIR"

echo "Run dir:    $RUN_DIR"
echo "Model:      $MODEL"
echo "Task/Diff:  $TASK / $DIFFICULTY (role=$ROLE)"
echo "Episodes:   $EPISODES   group_size=$GROUP_SIZE"
echo "Logs:       $RUN_DIR/train.log"
echo "Metrics:    $RUN_DIR/metrics.jsonl"
echo "TB:         $RUN_DIR/tb/   (tensorboard --logdir $RUN_DIR/tb --port 6006)"

python -u training/cyborg_grpo.py \
  --model "$MODEL" \
  --task "$TASK" --difficulty "$DIFFICULTY" --role "$ROLE" \
  --episodes "$EPISODES" --group-size "$GROUP_SIZE" \
  --rollout-mode episode \
  --output-dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/train.log"

echo
echo "Run done. Latest checkpoint: $RUN_DIR"
echo "Suggested next step:"
echo "  python training/eval_before_after.py \\"
echo "    --task $TASK --difficulty $DIFFICULTY --role $ROLE --episodes 32 \\"
echo "    --baselines random heuristic \\"
echo "    --base-model $MODEL \\"
echo "    --trained-model $RUN_DIR"
echo "  python training/plot_curves.py --metrics $RUN_DIR/metrics.jsonl"
