#!/usr/bin/env bash
# Provision a fresh DigitalOcean GPU droplet for CybOrg GRPO training.
#
# What it does, in order:
#   1. Sanity-check the GPU + driver + CUDA toolkit.
#   2. Install uv (fast Python package manager) and create a venv.
#   3. Install pinned ML stack: torch (CUDA wheel), transformers, trl,
#      accelerate, datasets, tensorboard, matplotlib.
#   4. Editable-install the local cyborg_env package and run the test suite.
#   5. Run the trainer for a 4-step smoke test so any failure surfaces *before*
#      you commit to a long run.
#
# Usage (on the droplet, after `git clone`):
#   bash scripts/droplet_bootstrap.sh
#
# After it succeeds, kick off a real run from inside tmux:
#   bash scripts/train.sh red_easy easy red 200 4
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

bold() { printf "\n\033[1m== %s ==\033[0m\n" "$*"; }

bold "1/5 GPU + driver + CUDA check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Pick a DigitalOcean GPU droplet image (Ubuntu 22.04 + NVIDIA drivers preinstalled)." >&2
  exit 1
fi
nvidia-smi | head -n 18
CUDA_MAJOR=$(nvidia-smi | awk -F'CUDA Version: ' '/CUDA Version/ {print $2}' | awk '{print int($1)}')
echo "Detected CUDA major: ${CUDA_MAJOR:-unknown}"

bold "2/5 Install uv + create venv (.venv)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python --version

bold "3/5 Install ML stack"
# H100 / A100 / L40S work with the cu121 wheels for torch>=2.3.
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url "$TORCH_INDEX" \
  "torch>=2.3,<2.6"
uv pip install \
  "transformers>=4.45,<5" \
  "trl>=0.11,<0.13" \
  "accelerate>=0.34" \
  "datasets>=2.20" \
  "tensorboard>=2.16" \
  "matplotlib>=3.8" \
  "pydantic>=2.7"

bold "4/5 Install cyborg_env (editable) + run tests"
uv pip install -e ./src
uv pip install pytest pytest-cov
pytest -q tests || { echo "test suite failed \u2014 abort"; exit 1; }

bold "5/5 4-step smoke test of the trainer"
mkdir -p runs assets
python training/cyborg_grpo.py --baseline-only --episodes 4 --task red_easy --role red
# Tiny real-training run: 4 prompts, group_size 2 \u2014 just proves the pipeline ends-to-end.
python training/cyborg_grpo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --task red_easy --difficulty easy --role red \
  --episodes 4 --group-size 2 \
  --output-dir runs/_smoke

echo
echo "\u2705 Bootstrap complete."
echo "   Activate the venv in new shells with:  source .venv/bin/activate"
echo "   Then start a real run from tmux:       bash scripts/train.sh red_easy easy red 200 4"
