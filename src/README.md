# CybOrg

> **Train an LLM to be a SOC analyst — and the attacker who has to dodge it.**

CybOrg is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant
**multi-agent cybersecurity reinforcement-learning environment** built for
the April 2026 Meta OpenEnv Hackathon.

A single environment exposes one of three role-conditioned tasks (Red,
Blue, or Dual self-play) on a procedurally-generated enterprise network.
The agent talks to it via plain JSON actions — no tensors, no graph
spaces, nothing the model has to learn to encode. Eight independent
reward columns (instead of one monolithic scalar) make reward hacking
hard and the trainer dashboard rich.

| Theme | Multi-Agent Interactions (Theme #1) — and a sprinkle of World Modeling (#3.1) |
|---|---|
| Action grammar | text-first JSON, validated by Pydantic |
| Reward shape | layered (`Sequential(format_gate, WeightedSum(...))`) |
| Difficulty | `easy` / `medium` / `hard`, selectable per `reset()` |
| Built-in opponent | scripted heuristic (training as red), reference attack trace (training as blue) |
| Storytelling demo | trained-vs-baseline transcripts in `assets/` |

---

## What you can train

```text
red_easy   →  train Red against a heuristic SOC analyst
blue_easy  →  train Blue against a deterministic attack trace
dual       →  alternate Red & Blue (self-play, two-policy, or shared params)
```

The same environment image serves all three tasks; pick the mode by
passing `task=<...>` to `reset()` (or via the `CYBORG_DEFAULT_TASK` env
var).

---

## Quickstart (local, no Docker)

```bash
git clone <this-repo>
cd cyborg/cyborg_env
uv sync                                  # one-time install
uv run --project . server --port 8000   # serve over HTTP/WS
```

In another shell, talk to it:

```python
import asyncio
from cyborg_env import CybOrg, CybOrgAction

async def main():
    async with CybOrg(base_url="http://localhost:8000") as env:
        r = await env.reset(seed=7, task="red_easy", difficulty="easy")
        r = await env.step(CybOrgAction(tool="scan",
                                        args={"target": "dmz-00"},
                                        role="red"))
        print(r.observation.last_action_result["stdout"])
        # -> "discovered dmz-00 (dmz) services=['ssh', 'rdp']"

asyncio.run(main())
```

Or run the in-process tests (32 of them):

```bash
cd cyborg_env && .venv/bin/python -m pytest ../tests
```

---

## Anatomy of a CybOrg episode

1. **`WorldGenerator(seed, difficulty)`** builds a deterministic
   network: 5–20 hosts across 2–4 zones, each running a sample of services
   from a real-world vulnerability template (`http`, `ssh`, `smb`, `rdp`,
   `kerberos`, plus local privilege-escalation flaws).

2. **Red** sees the network through a fog of war: only hosts it has
   scanned or pivoted into are visible. It earns reward for *new*
   accomplishments — owning a host, leaking a credential, crossing a
   zone, exfiltrating an asset.

3. **Blue** sees an alert stream — a mix of true positives raised by red
   actions and benign noise emitted by green NPC users. It earns reward
   for correctly attributing alerts and breaking attack paths, and is
   penalized for over-blocking (cutting innocent hosts off the network).

4. **Green NPCs** generate background traffic each turn, making the alert
   stream realistically noisy. Without them blue could win by isolating
   every host — *with* them, isolating wrong costs SLA continuity.

5. **The episode ends** when red exfiltrates the crown-jewel asset
   (red wins), or when the step budget runs out (blue wins by default).

---

## Reward design (eight independent columns)

We use OpenEnv's `Sequential(FormatGate, WeightedSum(...))` so:
* a malformed JSON action **gates the entire reward to zero** (no partial credit for cheating the verifier);
* every component of the score lands in its own column on the trainer dashboard.

**Red** rubric columns:

| Column | Weight | When it fires |
|---|---|---|
| `pwned_hosts` | +0.10 | per *new* host owned this episode |
| `obj_progress` | +0.20 | per *new* objective predicate satisfied |
| `exfil_value` | +0.50 × value/100 | first successful exfil per asset |
| `stealth_bonus` | +0.05 | turn produced zero new red-attributed alerts |
| `noise_penalty` | -0.10 | per detection event triggered by red |
| `claim_honesty` | -0.30 | reserved hook for false objective claims |
| `step_cost` | -0.01 | every turn |
| `terminal` | ±1 / -0.5 | win / loss / timeout |

**Blue** rubric columns:

| Column | Weight | When it fires |
|---|---|---|
| `detection_credit` | +0.15 | per new correctly-attributed alert |
| `containment_credit` | +0.30 | per new path broken before red exfils |
| `continuity` | -0.20 × frac | fraction of network isolated (SLA penalty) |
| `false_finding` | -0.20 | per `submit_finding` that targets a benign event |
| `over_block` | -0.15 | per block/isolate of a host with no red activity in last K turns |
| `step_cost` | -0.01 | every turn |
| `terminal` | ±1 / +0.5 | win / loss / timeout (blue gets a small survival bonus) |

### Why this is hard to hack

* **All scoring runs off the simulator's event log**, never off agent
  text. The model can't claim victory; only `EXFIL_SUCCESS` events
  count.
* The format gate is a `Sequential` head — a malformed action contributes
  *zero* reward, period.
* Each per-asset bonus fires once. Spam doesn't work.
* `block_ip`/`isolate_host` of a clean host triggers an `OVERBLOCK`
  event that subtracts more than the blocking action's potential
  credit, so flooding mitigations is net-negative.
* Action diversity is monitored — if >70% of recent actions share a
  fingerprint, a `diversity_warn` flag flips on the observation that
  trainers can pin to a separate penalty if needed.

---

## Repo layout

```
cyborg_env/                    # the OpenEnv package the Space deploys
  models.py                   # Pydantic action/observation/state types
  client.py                   # async EnvClient subclass
  openenv.yaml                # spec_version + runtime + env vars
  server/
    app.py                    # FastAPI app via openenv.create_app
    cyborg_env_environment.py # Environment subclass (reset/step/state)
    Dockerfile                # FROM openenv-base
    sim/
      world.py world.py generator.py events.py
      red_actions.py blue_actions.py green.py
      heuristic_blue.py reference_red.py
    rewards/
      shared.py red_rubric.py blue_rubric.py

tests/                         # 32 pytest cases (run in <2s)
training/
  cyborg_grpo.py              # TRL GRPO driver + RandomPolicy/HeuristicRedPolicy baselines
  notebook.ipynb              # Colab the judges run
PLAN.md                        # the design brief that drove every decision
```

---

## Showing improvement

Baselines (run on this repo, no GPU needed):

```text
$ python training/cyborg_grpo.py --baseline-only --episodes 32 --task red_easy --role red
baseline=random     episodes= 32 mean_reward=0.069
baseline=heuristic  episodes= 32 mean_reward=0.200
```

A trained policy should land somewhere between these two on first
contact and surpass the heuristic after a real GRPO run. The Colab in
`training/notebook.ipynb` produces a `assets/red_reward_curve.png` plot
that we link from the README results section once training completes.

---

## Compliance check (per OpenEnv judging guide)

- [x] Uses OpenEnv `Environment` base class
- [x] Uses OpenEnv `Action` / `Observation` / `State` Pydantic types
- [x] Uses OpenEnv `Rubric` composition (`Sequential(format_gate, WeightedSum)`)
- [x] Uses `create_app(EnvironmentClass, ActionClass, ObservationClass, env_name=...)`
- [x] Client extends `EnvClient[ActT, ObsT, StateT]` with the three required methods
- [x] `openenv.yaml` declares `spec_version: 1`, name, runtime, app, port, variables
- [x] Dockerfile FROM `openenv-base` with health check
- [x] No reserved tool names (`reset`, `step`, `state`, `close`)
- [x] `reset` accepts `seed` and `episode_id`
- [x] Observation has `done` (bool) and `reward` (float)
- [x] Step returns observation only (server enriches via base class)

---

## Links

* Hackathon plan & design rationale: [`../PLAN.md`](../PLAN.md)
* OpenEnv: <https://github.com/meta-pytorch/OpenEnv>
* TRL GRPO trainer: <https://huggingface.co/docs/trl/main/en/grpo_trainer>

---

> Built for the April 2026 OpenEnv Hackathon (India).
