"""Plot reward curves and the before/after bar chart for the README.

Reads two artifacts produced by other scripts and writes PNGs into
``assets/``:

    runs/<name>/metrics.jsonl   ->  assets/reward_curve.png
                                    assets/win_rate_curve.png
                                    assets/format_ok_curve.png
    assets/eval_results.json    ->  assets/before_after_reward.png
                                    assets/before_after_winrate.png

Usage:
    python training/plot_curves.py --metrics runs/red/metrics.jsonl
    python training/plot_curves.py --eval assets/eval_results.json
    python training/plot_curves.py            # both, using defaults
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        return matplotlib
    except ImportError as e:
        raise SystemExit("pip install matplotlib") from e


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _smooth(xs: List[float], window: int = 5) -> List[float]:
    if window <= 1 or len(xs) <= window:
        return xs
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def plot_training_curves(metrics_path: Path, out_dir: Path) -> List[Path]:
    if not metrics_path.exists():
        print(f"[plot] {metrics_path} not found, skipping training curves")
        return []
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    rows = _read_jsonl(metrics_path)
    if not rows:
        print(f"[plot] {metrics_path} is empty, skipping")
        return []

    steps = [r["step"] for r in rows]
    rewards = [r["reward_mean"] for r in rows]
    rewards_std = [r["reward_std"] for r in rows]
    win_red = [r["win_rate_red"] for r in rows]
    fmt_ok = [r["format_ok_rate"] for r in rows]

    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, rewards, color="#1f77b4", label="mean group reward", linewidth=1)
    ax.plot(steps, _smooth(rewards, 7), color="#1f77b4", linewidth=2.5,
            alpha=0.8, label="EMA(7)")
    ax.fill_between(
        steps,
        [m - s for m, s in zip(rewards, rewards_std)],
        [m + s for m, s in zip(rewards, rewards_std)],
        color="#1f77b4", alpha=0.15, label="\u00b11 std"
    )
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xlabel("training step"); ax.set_ylabel("reward")
    ax.set_title("CybOrg \u2014 GRPO mean group reward over training")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    p = out_dir / "reward_curve.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, win_red, label="red win rate", color="#d62728", linewidth=1)
    ax.plot(steps, _smooth(win_red, 7), color="#d62728", linewidth=2.5, alpha=0.8)
    ax.set_xlabel("training step"); ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    ax.set_title("CybOrg \u2014 red-side win rate over training")
    p = out_dir / "win_rate_curve.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, fmt_ok, label="format_ok rate", color="#2ca02c", linewidth=1)
    ax.plot(steps, _smooth(fmt_ok, 7), color="#2ca02c", linewidth=2.5, alpha=0.8)
    ax.set_xlabel("training step"); ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    ax.set_title("CybOrg \u2014 valid-JSON action rate over training")
    p = out_dir / "format_ok_curve.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    written.append(p)
    return written


def plot_before_after(eval_path: Path, out_dir: Path) -> List[Path]:
    if not eval_path.exists():
        print(f"[plot] {eval_path} not found, skipping before/after plots")
        return []
    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    payload = json.loads(eval_path.read_text())
    policies = payload["policies"]
    if not policies:
        return []
    labels = [p["label"] for p in policies]
    means = [p["reward_mean"] for p in policies]
    stds = [p["reward_std"] for p in policies]
    win_red = [p["win_rate_red"] for p in policies]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, means, yerr=stds, capsize=5,
                  color=["#9ecae1", "#fdae6b", "#a1d99b", "#fb6a4a"][: len(labels)])
    ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_ylabel("mean episode reward")
    ax.set_title(
        f"CybOrg \u2014 reward by policy "
        f"(task={payload['task']}, role={payload['role']}, n={payload['episodes']})"
    )
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{m:+.2f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    p = out_dir / "before_after_reward.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, win_red,
           color=["#9ecae1", "#fdae6b", "#a1d99b", "#fb6a4a"][: len(labels)])
    ax.set_ylim(0, 1.05); ax.set_ylabel("red win rate")
    ax.set_title(
        f"CybOrg \u2014 red win rate by policy "
        f"(task={payload['task']}, n={payload['episodes']})"
    )
    for i, v in enumerate(win_red):
        ax.text(i, v, f"{v:.0%}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    p = out_dir / "before_after_winrate.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    written.append(p)
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot training and eval artifacts")
    p.add_argument("--metrics", type=Path,
                   default=ROOT / "runs" / "red" / "metrics.jsonl")
    p.add_argument("--eval", type=Path, dest="eval_path",
                   default=ASSETS / "eval_results.json")
    p.add_argument("--out", type=Path, default=ASSETS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    written = []
    written.extend(plot_training_curves(args.metrics, args.out))
    written.extend(plot_before_after(args.eval_path, args.out))
    if not written:
        print("[plot] nothing produced \u2014 no metrics.jsonl or eval_results.json found.")
    else:
        print("[plot] wrote:")
        for p in written:
            print("  ", p)


if __name__ == "__main__":
    main()
