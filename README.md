# CybOrg — OpenEnv Hackathon Submission

A multi-agent **cybersecurity reinforcement-learning environment** built on
[OpenEnv](https://github.com/meta-pytorch/OpenEnv) for the April 2026 Meta
OpenEnv Hackathon. Train an LLM as the **Red** attacker, the **Blue**
defender, or both via self-play on a procedurally-generated enterprise
network — with eight independent reward columns that make reward hacking
hard and the trainer dashboard rich.

> **Theme**: Multi-Agent Interactions (with a sprinkle of World Modeling).

## Quick links

* **Story, quickstart, and reward design**: [`cyborg_env/README.md`](cyborg_env/README.md)
* **Architecture and engineering plan**: [`PLAN.md`](PLAN.md)
* **Run the tests**: `cd cyborg_env && uv run --project . python -m pytest ../tests`
* **Serve locally**: `cd cyborg_env && uv run --project . server --port 8000`
* **Train baselines**:
  `python training/cyborg_grpo.py --baseline-only --episodes 32 --task red_easy --role red`

## Hackathon Judging Criteria

| Criterion | Where to look |
|---|---|
| **Innovation (40%)** — multi-agent + procedural worlds + eight reward columns + format gate | [`cyborg_env/README.md` — Reward design](cyborg_env/README.md#reward-design-eight-independent-columns) |
| **Storytelling (30%)** — narrative arc + demo transcripts | [`cyborg_env/README.md` — Anatomy of an episode](cyborg_env/README.md#anatomy-of-a-cyborg-episode) and `assets/` |
| **Reward Improvement (20%)** — baseline numbers + Colab + curve | [`training/notebook.ipynb`](training/notebook.ipynb) |
| **Reward / Pipeline Setup (10%)** — OpenEnv compliance, Dockerfile, manifest, tests | [`cyborg_env/README.md` — Compliance check](cyborg_env/README.md#compliance-check-per-openenv-judging-guide) |
