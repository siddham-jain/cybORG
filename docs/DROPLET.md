# Training CybOrg on a DigitalOcean GPU droplet

Step-by-step runbook to take this repo from "checked out" to "publishable
reward curve + before/after eval" on a single DigitalOcean GPU droplet.
Everything below is scripted in `scripts/`; this doc just narrates the
commands and where to look when things go wrong.

The artifacts you will produce satisfy the **Showing Improvement in
Rewards (20%)** judging criterion:

| Artifact | What it proves | Produced by |
|---|---|---|
| `runs/<id>/metrics.jsonl` | per-step mean reward, std, win rate, format-ok rate, sample completion | trainer callback |
| `runs/<id>/tb/` | TensorBoard event files | TRL via `report_to=["tensorboard"]` |
| `assets/reward_curve.png` | reward curve (mean ± std, EMA-smoothed) | `training/plot_curves.py` |
| `assets/win_rate_curve.png` | red win-rate climbing during training | `training/plot_curves.py` |
| `assets/format_ok_curve.png` | JSON-validity rate climbing during training | `training/plot_curves.py` |
| `assets/eval_results.json` | held-out N-seed comparison: random vs heuristic vs LLM-base vs LLM-trained | `training/eval_before_after.py` |
| `assets/before_after_reward.png` | bar chart of the four policies' mean reward | `training/plot_curves.py` |
| `assets/transcripts/<label>_seed_*.md` | full episode logs you can read end-to-end | `training/eval_before_after.py` |

---

## 0. Local sanity check (do this first, no GPU needed)

Before you spin up a paid droplet, prove the import path and reward
function still work on your Mac:

```bash
# from the repo root
PYTHONPATH=src python3 training/cyborg_grpo.py --baseline-only --episodes 8 --task red_easy --role red
```

If you see `baseline=random ... mean_reward=...` and
`baseline=heuristic ... mean_reward=...`, you're good. Skip this only if
you trust the repo.

---

## 1. Pick a droplet

| Tier | Droplet | When to use | ~Cost |
|---|---|---|---|
| Cheap | RTX 6000 Ada (48 GB) — `gpu-rtx6000ada-1x` | Qwen 0.5B–3B Instruct, fast iteration | $0.76/hr |
| Real | H100 ×1 (80 GB) — `gpu-h100x1-80gb` | Qwen 7B / Llama 3.1 8B, bf16 GRPO | $4.89/hr |
| Big | H100 ×8 (640 GB) — `gpu-h100x8-640gb` | 13B+ models, multi-GPU GRPO with `accelerate` | $39/hr |

For an OpenEnv hackathon submission with reward curves to show, the
**RTX 6000 Ada** is the right starting point. Train a 0.5B for 1–2 hours
and the curves will be informative.

When creating the droplet:
- **Image**: "AI/ML Ready" (Ubuntu 22.04 with NVIDIA driver + CUDA preinstalled)
- **Region**: any with GPU stock; NYC2/SFO3/AMS3 are reliable
- **SSH key**: paste your public key (don't use password auth)
- **VPC**: default
- **Backups / monitoring**: optional, nothing in this run requires them

DigitalOcean's GPU droplets ship with the NVIDIA driver and CUDA toolkit
already installed on the AI/ML image, which is what
`scripts/droplet_bootstrap.sh` assumes.

---

## 2. SSH in and start a tmux session

`tmux` is non-negotiable: SSH disconnects mid-training are normal, and
a detached session keeps the run alive.

```bash
ssh root@<droplet-ip>

apt update && apt install -y tmux htop nvtop git curl
tmux new -s cyborg     # create a session named 'cyborg'
# (later, reattach with:  tmux attach -t cyborg )
```

Inside the tmux session:

```bash
git clone https://github.com/<your-fork>/cybORG.git
cd cybORG
bash scripts/droplet_bootstrap.sh
```

The bootstrap script:
1. Verifies `nvidia-smi` works.
2. Installs `uv` and creates `.venv` (Python 3.11).
3. Installs torch (CUDA wheels), transformers, trl, accelerate, datasets,
   tensorboard, matplotlib, pydantic.
4. Installs `cyborg_env` from `./src` in editable mode.
5. Runs the test suite (`pytest -q tests`).
6. Runs a 4-step trainer smoke test so the full pipeline is exercised
   *before* you start a long run.

If step 6 produces a `runs/_smoke/metrics.jsonl` with at least one row,
you're done bootstrapping.

---

## 3. Kick off the real training run

From the same tmux session:

```bash
source .venv/bin/activate
bash scripts/train.sh red_easy easy red 200 4 Qwen/Qwen2.5-0.5B-Instruct
```

That writes everything under
`runs/red-red_easy-easy-YYYYMMDD-HHMMSS/`:
- `train.log` — full stdout/stderr (tee'd by the wrapper)
- `metrics.jsonl` — one JSON object per training step
- `tb/` — TensorBoard event files
- `checkpoint-N/` — model weights (auto-saved every ~50 steps, last 3 kept)

Detach from tmux with `Ctrl-b d` and the run keeps going.

---

## 4. Monitor while it runs

### Quick console glance (single command)

```bash
bash scripts/monitor.sh                 # auto-picks the latest run dir
```

This opens a 4-pane tmux window with:
- live `nvidia-smi` (GPU utilization, memory, temp, power)
- `tail -f train.log` (everything the trainer prints)
- `tail -f metrics.jsonl` (one row per training step)
- `htop` (CPU + RAM)

Detach with `Ctrl-b d`. Reattach with `tmux attach -t cyborg-monitor`.

### TensorBoard from your laptop

Forward the droplet's port 6006 over SSH:

```bash
# on your laptop, in a separate terminal:
ssh -N -L 6006:localhost:6006 root@<droplet-ip>
```

Then on the droplet (in the cyborg tmux session, new window with `Ctrl-b c`):

```bash
source .venv/bin/activate
tensorboard --logdir runs --port 6006 --host 0.0.0.0
```

Open <http://localhost:6006> on your laptop. You'll see `train/loss`,
`train/reward`, `train/kl`, and the per-component reward decomposition
that TRL emits.

### What "healthy" training looks like

In `metrics.jsonl` you should see, within ~50 steps:
- `reward_mean` trending **up** (it can be noisy; smooth with EMA).
- `format_ok_rate` climbing toward `1.0` (the model learns valid JSON).
- `win_rate_red` ticking up from baseline (random ~0–5%, heuristic ~30–60%).
- `mean_episode_steps` should grow if you trained with `--rollout-mode episode`.

### What "things are wrong" looks like

| Symptom | Likely cause | Fix |
|---|---|---|
| `reward_mean` is flat at exactly 0.0 across many steps | Format gate killing every action (model emits prose) | Lower `--learning-rate` or shorten the system prompt; sample with `temperature` |
| `nvidia-smi` shows 0% util but the run is alive | Trainer is generating, not stepping; or batch is too small | Bump `--group-size`; raise `--max-completion-length` |
| OOM mid-step | Model + batch too big for the card | Drop to a smaller model, or reduce `--group-size`, `--max-prompt-length` |
| Loss is `nan` after a few steps | bf16 instability with this model | Re-run with `--no-bf16` (slower but safe) |
| `format_ok_rate` stuck at 0 | System prompt formatting confuses the model | Inspect `metrics.jsonl[*].sample_completion` and tweak `SYSTEM_PROMPT_RED` |

You can read sample completions live by running:

```bash
jq -r '.sample_completion' runs/<run-dir>/metrics.jsonl | tail -n 20
```

---

## 5. After training: produce the evidence artifacts

Once `train.sh` exits (or you stop it with `Ctrl-c` in its tmux pane),
run the eval and the plotter:

```bash
RUN_DIR=$(ls -dt runs/*/ | head -n1)

python training/eval_before_after.py \
  --task red_easy --difficulty easy --role red --episodes 32 \
  --baselines random heuristic \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --trained-model "$RUN_DIR"

python training/plot_curves.py --metrics "$RUN_DIR/metrics.jsonl"
```

You will now have:

```
assets/eval_results.json
assets/reward_curve.png
assets/win_rate_curve.png
assets/format_ok_curve.png
assets/before_after_reward.png
assets/before_after_winrate.png
assets/transcripts/random_seed_000.md
assets/transcripts/heuristic_seed_000.md
assets/transcripts/llm_base_seed_000.md
assets/transcripts/llm_trained_seed_000.md
```

These are the deliverables you cite in `cyborg_env/README.md` under
"Reward Improvement".

---

## 6. Pull artifacts back to your laptop

```bash
# on your laptop
mkdir -p ./pulled
scp -r root@<droplet-ip>:~/cybORG/assets ./pulled/assets
scp -r root@<droplet-ip>:~/cybORG/runs/<latest> ./pulled/runs
```

Commit `assets/` to your repo. Leave `runs/` out of git (the JSONL is
small enough to keep, but checkpoints are large — add them to
`.gitignore`).

---

## 7. Tear the droplet down

Don't forget. GPU droplets bill by the hour.

```bash
doctl compute droplet delete <droplet-id>
# or via the DO web UI: Droplets -> ... -> Destroy
```

---

## Cheat sheet

```bash
# bootstrap once
bash scripts/droplet_bootstrap.sh

# train (defaults shown)
bash scripts/train.sh red_easy easy red 200 4 Qwen/Qwen2.5-0.5B-Instruct

# watch
bash scripts/monitor.sh
ssh -N -L 6006:localhost:6006 root@<ip>   # then `tensorboard --logdir runs --port 6006`

# evidence
python training/eval_before_after.py --episodes 32 \
  --base-model Qwen/Qwen2.5-0.5B-Instruct --trained-model runs/<latest>
python training/plot_curves.py --metrics runs/<latest>/metrics.jsonl

# ship
scp -r root@<ip>:~/cybORG/assets ./assets
```
