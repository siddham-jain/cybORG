"""Before/after evaluation for the CybOrg GRPO submission.

Runs N held-out seeds against:
    1. one or more *baselines* (random / heuristic / a base HF model), and
    2. a *trained* checkpoint produced by ``training/cyborg_grpo.py``,
records per-episode reward, win rate, mean episode length, format-error
rate, and writes:

    assets/eval_results.json                 # summary table for the README
    assets/transcripts/<label>_seed_<N>.md   # human-readable game logs

The README and the slides cite the JSON; the transcripts are the
qualitative "before vs after" evidence judges look for under the
"Showing Improvement in Rewards (20%)" criterion.

Quick usage (no GPU required):
    python training/eval_before_after.py \
        --task red_easy --difficulty easy --role red --episodes 16 \
        --baselines random heuristic

With a trained checkpoint:
    python training/eval_before_after.py \
        --task red_easy --difficulty easy --role red --episodes 32 \
        --baselines random heuristic \
        --base-model Qwen/Qwen2.5-0.5B-Instruct \
        --trained-model ./runs/red
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
ASSETS = ROOT / "assets"

# Make the repo root importable so ``training.cyborg_grpo`` resolves whether
# the user runs this as ``python training/eval_before_after.py`` or as a
# module. Mirrors the same trick used in ``cyborg_grpo.py``.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from cyborg_env.server.cyborg_env_environment import CybOrgEnvironment  # noqa: F401
except ImportError:
    sys.path.insert(0, str(SRC))
    from server.cyborg_env_environment import CybOrgEnvironment  # type: ignore[no-redef]  # noqa: F401

from training.cyborg_grpo import (  # type: ignore[import]
    HeuristicRedPolicy,
    RandomPolicy,
    RolloutResult,
    SYSTEM_PROMPT_BLUE,
    SYSTEM_PROMPT_RED,
    _build_user_prompt,
    _parse_action,
    rollout_episode,
)


@dataclass
class PolicyMetrics:
    label: str
    episodes: int
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    win_rate_red: float
    win_rate_blue: float
    timeout_rate: float
    mean_episode_steps: float
    format_ok_rate: float
    per_seed: List[Dict[str, Any]] = field(default_factory=list)


def summarise(label: str, results: List[RolloutResult]) -> PolicyMetrics:
    rewards = [r.episode_reward for r in results]
    steps = [r.steps for r in results]
    wins_red = sum(1 for r in results if r.winner == "red")
    wins_blue = sum(1 for r in results if r.winner == "blue")
    timeouts = sum(1 for r in results if r.winner not in {"red", "blue"})
    fmt_rates = [
        (sum(1 for f in r.format_oks if f) / max(1, len(r.format_oks)))
        for r in results
    ]
    n = max(1, len(results))
    return PolicyMetrics(
        label=label,
        episodes=len(results),
        reward_mean=statistics.mean(rewards) if rewards else 0.0,
        reward_std=statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        reward_min=min(rewards) if rewards else 0.0,
        reward_max=max(rewards) if rewards else 0.0,
        win_rate_red=wins_red / n,
        win_rate_blue=wins_blue / n,
        timeout_rate=timeouts / n,
        mean_episode_steps=statistics.mean(steps) if steps else 0.0,
        format_ok_rate=statistics.mean(fmt_rates) if fmt_rates else 0.0,
        per_seed=[
            {
                "seed": i,
                "reward": r.episode_reward,
                "steps": r.steps,
                "winner": r.winner,
                "format_ok_rate": fmt_rates[i],
            }
            for i, r in enumerate(results)
        ],
    )


def write_transcript(out_dir: Path, label: str, seed: int, role: str, result: RolloutResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    f = out_dir / f"{label}_seed_{seed:03d}.md"
    lines: List[str] = []
    lines.append(f"# CybOrg episode \u2014 policy={label}, role={role}, seed={seed}")
    lines.append("")
    lines.append(f"- episode reward: **{result.episode_reward:+.3f}**")
    lines.append(f"- winner: **{result.winner or 'timeout'}**")
    lines.append(f"- steps: **{result.steps}**")
    lines.append("")
    for i, (completion, reward, fmt_ok, breakdown) in enumerate(
        zip(result.completions, result.rewards, result.format_oks, result.score_breakdowns)
    ):
        lines.append(f"## Turn {i+1}")
        lines.append(f"reward `{reward:+.3f}` format_ok `{fmt_ok}`")
        if breakdown:
            top = sorted(breakdown.items(), key=lambda kv: -abs(kv[1]))[:5]
            lines.append("top components: " + ", ".join(f"`{k}={v:+.2f}`" for k, v in top))
        lines.append("```")
        lines.append(completion.strip()[:600])
        lines.append("```")
        lines.append("")
    f.write_text("\n".join(lines))


class HFPolicy:
    """Wraps a Hugging Face causal LM as a `policy.generate(prompt, ...)`."""

    def __init__(self, model_name_or_path: str, dtype: str = "auto", device: Optional[str] = None):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise SystemExit("pip install torch transformers") from e
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = (
            torch.bfloat16 if dtype == "bf16"
            else torch.float16 if dtype == "fp16"
            else "auto"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self._torch = torch

    def generate(self, prompt: str, max_new_tokens: int = 96, tokenizer=None) -> str:
        with self._torch.inference_mode():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def evaluate_policy(
    label: str,
    policy,
    *,
    episodes: int,
    task: str,
    difficulty: str,
    role: str,
    transcripts_per_policy: int,
    transcripts_dir: Path,
) -> PolicyMetrics:
    env = CybOrgEnvironment()
    results: List[RolloutResult] = []
    for seed in range(episodes):
        r = rollout_episode(
            env, seed=seed, task=task, difficulty=difficulty, role=role,
            policy=policy, tokenizer=None,
        )
        results.append(r)
        if seed < transcripts_per_policy:
            write_transcript(transcripts_dir, label, seed, role, r)
    metrics = summarise(label, results)
    print(
        f"[eval] {label:>14s} episodes={metrics.episodes:3d} "
        f"reward={metrics.reward_mean:+.3f}\u00b1{metrics.reward_std:.3f} "
        f"win_red={metrics.win_rate_red:.2f} "
        f"format_ok={metrics.format_ok_rate:.2f} "
        f"steps={metrics.mean_episode_steps:.1f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Before/after eval for CybOrg GRPO")
    p.add_argument("--task", choices=["red_easy", "blue_easy", "dual"], default="red_easy")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    p.add_argument("--role", choices=["red", "blue"], default="red")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument(
        "--baselines",
        nargs="*",
        default=["random", "heuristic"],
        choices=["random", "heuristic"],
        help="Reference policies to evaluate alongside any LLM checkpoints.",
    )
    p.add_argument("--base-model", type=str, default=None,
                   help="HF model id evaluated as the *untrained* baseline.")
    p.add_argument("--trained-model", type=str, default=None,
                   help="Path to the GRPO checkpoint produced by training.")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--transcripts-per-policy", type=int, default=3)
    p.add_argument("--out", type=str, default=str(ASSETS / "eval_results.json"))
    p.add_argument("--transcripts-dir", type=str, default=str(ASSETS / "transcripts"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    transcripts_dir = Path(args.transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    all_metrics: List[PolicyMetrics] = []
    for name in args.baselines:
        policy = RandomPolicy(seed=0) if name == "random" else HeuristicRedPolicy()
        all_metrics.append(evaluate_policy(
            label=name, policy=policy, episodes=args.episodes, task=args.task,
            difficulty=args.difficulty, role=args.role,
            transcripts_per_policy=args.transcripts_per_policy,
            transcripts_dir=transcripts_dir,
        ))

    if args.base_model:
        all_metrics.append(evaluate_policy(
            label="llm_base", policy=HFPolicy(args.base_model, dtype=args.dtype),
            episodes=args.episodes, task=args.task, difficulty=args.difficulty,
            role=args.role, transcripts_per_policy=args.transcripts_per_policy,
            transcripts_dir=transcripts_dir,
        ))
    if args.trained_model:
        all_metrics.append(evaluate_policy(
            label="llm_trained", policy=HFPolicy(args.trained_model, dtype=args.dtype),
            episodes=args.episodes, task=args.task, difficulty=args.difficulty,
            role=args.role, transcripts_per_policy=args.transcripts_per_policy,
            transcripts_dir=transcripts_dir,
        ))

    payload = {
        "task": args.task,
        "difficulty": args.difficulty,
        "role": args.role,
        "episodes": args.episodes,
        "policies": [asdict(m) for m in all_metrics],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")
    print(f"transcripts: {transcripts_dir}")


if __name__ == "__main__":
    main()
