# CybOrg — Plan

## TL;DR

CybOrg is an OpenEnv-compliant **multi-agent cybersecurity reinforcement
learning environment** designed to train LLMs to think like attackers,
defenders, and the noisy users who get caught in between. A single environment
exposes one of three role-conditioned tasks (Red, Blue, or Dual) on a
procedurally-generated enterprise network, with verifiable, layered rewards and
a built-in opponent that scales with the agent.

This document is the design brief that drives every implementation decision.

---

## 1. Analysis of reference work

### 1.1 What each existing repo does well

**open-range (open-cybernauts)**
- *World model*: rich manifest-first compiler (`WorldIR`) with zones, services,
  workflows, NPC profiles, weakness families, and admission probes that prove a
  world is trainable.
- *Reward shape*: terminal-first (winner ±1) with **shaped milestone bumps** for
  red (`InitialAccess`, `CredentialObtained`, `CrossZoneTraversal`,
  `SensitiveAssetRead`) and blue (detection, containment, continuity penalty).
- *Multi-agent*: explicit red / blue / green roles, with `red_only`,
  `blue_only_live`, `blue_only_from_prefix`, `joint_pool` modes. `next_decision`
  yields whichever externally controlled actor is due.
- *Anti-hack*: hallucination penalty for false `claim_objective`, action
  diversity audit, suspicious-action collapse warning, integrity probes.

**PenGym (cyb3rlab)**
- Wraps the well-known **NASim** action space (Service / OS / Subnet /
  Exploit / Privilege-escalation) on top of a real CyRIS-built KVM cyber range.
- Demonstrates the *gap* between a clean discrete action space and gritty
  exploit execution — useful as a reminder that what trains an LLM is the
  *interface*, not the underlying VMs.
- Network/host vector representation is compact and easy to verify.

**CyberBattleSim (Microsoft)**
- *Node / Vulnerability / Credential* model: every node has services, firewall
  rules, properties, and vulnerabilities with `precondition` (boolean
  expression), `outcome` (LeakedCredentials / LeakedNodesId / LateralMove /
  PrivilegeEscalation / CustomerData), and `Rates` (probing/exploit detection,
  success).
- Action space: `local_vulnerability`, `remote_vulnerability`, `connect`.
- Defender agent abstraction with reimaging cost.
- Continuous reward shaped by node value, credential discovery, lateral
  movement; final victory when all nodes owned.

### 1.2 Where each falls short for LLM training

| Concern | open-range | PenGym | CyberBattleSim |
|---|---|---|---|
| LLM-friendly text I/O | partial | no, structured numeric vectors | no, dict of int spaces |
| Runs in <1 GPU-min | needs Kind | needs full cyber range | yes |
| Procedural difficulty curve | yes (manifests) | scenario-only | weak (one fixed network/sample) |
| Self-play dual-role | yes but heavy | no | no |
| Reward hack resistant | yes (audit) | n/a | weak (shape exploitable) |
| Hosted on HF Spaces | no | no | no |
| Built-in opponent that scales | no | no | scripted |

**The opportunity**: combine open-range's *role-aware terminal+shaped reward* +
*green NPC noise* with CyberBattleSim's *clean credential/vulnerability graph
model* and ship it as an LLM-native, text-prompted, one-container OpenEnv that
can train both sides on the same world.

### 1.3 Best ideas we adopt

1. **Manifest → procedural world** (open-range) so we can generate an unbounded
   curriculum of admitted snapshots, not one fixed CTF.
2. **Credential / vulnerability / firewall graph** (CyberBattleSim) so red
   actions have crisp preconditions and post-conditions a verifier can check.
3. **Milestone shaped reward + terminal verdict + continuity penalty**
   (open-range) so RL doesn't stall on sparse signal but can't be easily hacked.
4. **Green NPC traffic** (open-range) to give blue meaningful noise to filter,
   and so red can blend in.
5. **Role-conditioned single env exposing 3 modes** so a single Hugging Face
   Space supports three task variants without three deploys.
6. **OpenEnv `Rubric` composition** so we expose multiple independent reward
   columns (`pwned`, `exfiltrated`, `noisy`, `format_ok`, `objective_claim_ok`,
   `continuity`, `detected`, `contained`) that judges and trainers can both
   monitor.

---

## 2. CybOrg system overview

### 2.1 What the agent sees and does (text-first, LLM-native)

CybOrg is a text-action environment. The model receives a JSON-structured
observation and emits a JSON action — that is the entire interface. No tensors,
no graph spaces, nothing the model has to learn to encode. This is by design,
because:
- LLM RL with TRL/GRPO works best on text completions.
- It makes the verifier and the rubric trivially auditable.
- It works on any base/instruct model with no SFT.

**Action grammar** (single Pydantic union type, validated on each step):

```json
{ "tool": "scan",          "args": {"target": "<host>"} }
{ "tool": "exploit",       "args": {"target": "<host>", "vuln": "<id>"} }
{ "tool": "use_credential","args": {"cred": "<id>", "target": "<host>", "service": "<svc>"} }
{ "tool": "lateral_move",  "args": {"from": "<host>", "to": "<host>"} }
{ "tool": "exfiltrate",    "args": {"target": "<host>", "asset": "<id>"} }
{ "tool": "phish",         "args": {"role": "<sales|engineer|finance|it_admin>"} }
{ "tool": "wait",          "args": {} }
```

Blue actions:
```json
{ "tool": "monitor",        "args": {"host": "<host>"} }
{ "tool": "analyze_alert",  "args": {"alert_id": "<id>"} }
{ "tool": "block_ip",       "args": {"host": "<host>"} }
{ "tool": "isolate_host",   "args": {"host": "<host>"} }
{ "tool": "patch",          "args": {"host": "<host>", "vuln": "<id>"} }
{ "tool": "rotate_cred",    "args": {"cred": "<id>"} }
{ "tool": "submit_finding", "args": {"host": "<host>", "vuln": "<id>", "actor": "red"} }
{ "tool": "wait",           "args": {} }
```

**Observation grammar** (role-aware, partially observable):

```json
{
  "role": "red",
  "turn": 7,
  "discovered_hosts": [ {"id":"web-01","zone":"dmz","services":["http"], "owned":false} ],
  "credentials": [ {"id":"web_user","source":"web-01"} ],
  "objectives": ["read finance_docs", "obtain idp_admin_cred"],
  "objectives_done": ["initial_access@web-01"],
  "last_action_result": {"ok": true, "stdout": "...", "stderr": ""},
  "alerts_visible_to_blue": null,
  "score_breakdown": {"milestones": 0.2, "exfil_progress": 0.0, "noise_penalty": -0.05},
  "budget": {"steps_left": 23, "noise_left": 6}
}
```

Blue sees `alerts`, `service_health`, `green_baseline`, NOT the red action
log. Red sees `discovered_hosts`, `credentials`, NOT blue's findings list.

### 2.2 World model

The world is a directed graph of hosts:

```
Host: id, zone, services[], firewall_rules, vulnerabilities[], data_assets[],
      privilege_level (per role), owned, isolated, patched_vulns
Vulnerability: id, type (REMOTE|LOCAL), precondition, outcome, success_rate,
               detection_rate
Credential: id, valid_for (host/service/role), expired
Asset: id, value (1..100), class ("crown_jewel"|"sensitive"|"ordinary")
GreenNPC: id, role, awareness, susceptibility, routine
```

Worlds are produced by `WorldGenerator(seed, difficulty)` from a small set of
parametric templates (Tier-1 startup, Tier-2 corporate, Tier-3 enterprise).
This gives RLVE-style **adaptive procedural difficulty** out of the box.

### 2.3 Three tasks (the difficulty curve)

| Task        | Mode       | What model controls          | Built-in opponent              |
|-------------|------------|------------------------------|--------------------------------|
| `red_easy`  | red_only   | Red                          | Heuristic blue + light green   |
| `blue_easy` | blue_only  | Blue                         | Scripted reference red trace   |
| `dual`      | joint      | Red AND Blue alternating     | None — self-play / two-policy  |

Each task accepts `difficulty: easy|medium|hard` which controls:
- network size (5 / 10 / 20 hosts),
- vuln density and detection rates,
- green noise rate,
- step budget (30 / 50 / 80),
- presence of advanced asset (`payroll_db`) and zone count.

Difficulty is **selected per `reset()`** — the trainer can curriculum on this.

### 2.4 Reward design (multi-component, hard to hack)

We use OpenEnv's `Rubric` system to expose **independently monitored**
components. Final per-step reward = weighted sum + terminal bonus.

**Red rubric** (training a red agent):
- `pwned_hosts`: +0.10 per *new* host owned (capped). [milestone shaping]
- `obj_progress`: +0.20 per *new* objective predicate satisfied. [milestone]
- `exfil_value`: +0.50 × (asset.value / 100) on first successful exfil per asset.
- `stealth_bonus`: +0.05 per step with zero new alerts raised against red.
- `noise_penalty`: -0.10 per detection event triggered by red action.
- `format_ok`: 0.0 if action JSON invalid (gate, episode step skipped).
- `claim_honesty`: -0.30 if action contains `claim_objective` not satisfied.
- `step_cost`: -0.01 per step.
- **Terminal**: +1.0 win / -1.0 loss / -0.5 timeout.

**Blue rubric** (training a blue agent):
- `detection_credit`: +0.15 per *new* malicious event correctly attributed.
- `containment_credit`: +0.30 per *new* path broken before red completes.
- `continuity`: -0.20 × continuity_drop (false-positive isolations cost SLA).
- `false_finding`: -0.20 per submitted finding that targets a benign event.
- `over_block`: -0.15 per `block_ip`/`isolate_host` of a host with no malicious
  activity in last K turns.
- `format_ok`: 0.0 if invalid (gate).
- `step_cost`: -0.01 per step.
- **Terminal**: +1.0 if red prevented from reaching crown_jewel within budget.

**Dual rubric** zips both — each side gets its own reward column, and the
trainer can train Red, Blue, or both with shared parameters.

Reward-hack defenses:
- Action diversity audit (collapse warning if >70% identical fingerprints).
- `Sequential(format_gate, ...)` ensures malformed JSON yields 0 + skipped step.
- `claim_honesty` penalty (open-range borrowing) blocks "I won, trust me" hacks.
- Asset-value bonus only fires once per asset per episode.
- `block_ip` of every host in the network is penalized (false-positive flood).
- All scoring is computed from **simulator-emitted events**, never from agent
  text claims.

### 2.5 Built-in opponent (so the model isn't training on a vacuum)

When the model controls Red, Blue runs as a scripted heuristic that:
1. Monitors a random subset of hosts each turn.
2. Submits a finding if a `SuspiciousActionObserved` event is in its window.
3. Isolates a host once it sees ≥2 alerts on that host.
4. Patches a known-vulnerable host probabilistically.

When the model controls Blue, Red follows a **reference attack trace** built at
world-generation time (the trace is admission-proven to work). This guarantees
positive learning signal from step 1 — **no sparse-reward death**.

### 2.6 Anti-hacking checklist (preempt judges' questions)

- All rewards derived from internal events, not parsed from completion text.
- Action JSON validation gates the step (no reward at all if malformed).
- Hallucination penalty for explicit false "claim_objective".
- Blue cannot win by isolating everything (continuity penalty).
- Red cannot win by spamming actions (step budget + noise penalty + diversity audit).
- Per-episode action cap prevents runaway loops.
- All randomness is seeded per episode for reproducible verification.

---

## 3. Why this should win

| Criterion (weight) | How CybOrg scores |
|---|---|
| **Innovation (40%)** | Multi-agent cyber range with procedural difficulty, three-mode role-conditioned tasks, a real adversarial opponent built in, and event-grounded LLM-native action grammar. Cybersecurity is *underexplored* in LLM RL, and a defender that learns to triage real alert streams is a genuinely new capability. |
| **Storytelling (30%)** | Concrete narrative — "we trained an LLM to be a SOC analyst and to be the attacker who has to dodge it." Demo will show baseline vs trained on the same world, with role transcripts. README will frame it for a non-technical reader. |
| **Reward improvement (20%)** | Shaped + dense + verifier-ready means GRPO actually moves the needle in <100 steps on small models. We commit to a real Colab run with reward curves checked into the repo. |
| **Reward / pipeline (10%)** | Eight independent rubric columns expose every signal the trainer cares about; format-gate + claim-honesty + diversity-audit show we thought about reward hacking. Wordle-style training script ships in repo. |

---

## 4. Engineering plan

### 4.1 Repo layout (OpenEnv standard)

```
cyborg_env/
  __init__.py                # re-exports CybOrgEnv, CybOrgAction, CybOrgObservation
  client.py                  # EnvClient subclass
  models.py                  # Action / Observation / State pydantic models
  openenv.yaml               # spec_version, name, runtime, app, port, variables
  pyproject.toml             # depends on openenv-core
  README.md                  # the storytelling artifact (judges read this)
  uv.lock
  server/
    __init__.py
    app.py                   # create_app(...) glue
    cyborg_environment.py    # Environment subclass (reset/step/state)
    Dockerfile               # FROM openenv-base
    requirements.txt
    sim/                     # the actual world simulator
      __init__.py
      world.py               # World, Host, Vulnerability, Credential, Asset
      generator.py           # WorldGenerator(seed, difficulty)
      events.py              # EventLog, EventType
      red_actions.py
      blue_actions.py
      green.py               # NPC traffic
      heuristic_blue.py
      reference_red.py       # deterministic scripted attacker
    rewards/
      __init__.py
      red_rubric.py          # WeightedSum(...) of independent Rubrics
      blue_rubric.py
      shared.py              # FormatGate, StepCost, etc.
training/
  cyborg_grpo.py             # Wordle-style TRL GRPO script (Red task)
  cyborg_grpo_blue.py        # same for Blue
  notebook.ipynb             # Colab the judges run
tests/
  test_world.py
  test_red_actions.py
  test_blue_actions.py
  test_episode.py
  test_rewards.py
  test_format_gate.py
  test_openenv_compliance.py
PLAN.md
README.md
```

### 4.2 Build order (do first → do last)

1. **`models.py`** — exact Pydantic shapes for action/observation. Validate hard.
2. **`sim/world.py` + `sim/generator.py`** — deterministic procedural world.
3. **`sim/events.py` + `red_actions.py` + `blue_actions.py`** — verifier and
   action handlers. No reward yet.
4. **`sim/heuristic_blue.py` + `sim/reference_red.py`** — built-in opponents.
5. **`rewards/`** — rubrics with all eight columns and the format gate.
6. **`server/cyborg_environment.py`** — wire up reset/step/state and switch on
   task mode (`red_easy` / `blue_easy` / `dual`).
7. **`server/app.py` + `Dockerfile` + `openenv.yaml`** — package as OpenEnv.
8. **`client.py`** — friendly client + helper methods (`step_red`, `step_blue`).
9. **Tests** — episode-rollout sanity, reward independence, format gate, anti-hack.
10. **`training/cyborg_grpo.py`** — Wordle-style script using `generate_rollout_completions`.
11. **`README.md`** — story + plots + hosted-Space link.

### 4.3 Compliance checklist

- Uses OpenEnv `Environment` base class.
- Uses OpenEnv `Action`, `Observation`, `State` types.
- Uses OpenEnv `Rubric` composition.
- Uses `create_app(EnvironmentClass, ActionClass, ObservationClass, env_name=...)`.
- Client extends `EnvClient[ActT, ObsT, StateT]` with the three required methods.
- `openenv.yaml` declares `spec_version: 1`, `name`, `runtime: fastapi`,
  `app: server.app:app`, `port: 8000`, plus a `variables:` block for default task.
- Dockerfile FROM `openenv-base` with health check.
- No reserved tool names (`reset`, `step`, `state`, `close`).
- Reset accepts `seed` and `episode_id`.
- Observation has `done` (bool) and `reward` (float).
- Step returns observation only (server enriches via the Environment base class).

### 4.4 Risk register

| Risk | Mitigation |
|---|---|
| World too complex → reward signal too sparse | Default to `easy`, use built-in opponent, milestone shaping. |
| Reward hacking via JSON tricks | Strict Pydantic schema + format gate + diversity audit. |
| LLM never produces valid JSON | System prompt provides exact schema and one-shot example. |
| Blue cannot win on `easy` | Reference red is *intentionally noisy* on easy (high detection rates). |
| Episode runs forever | Hard step budget per difficulty. |
| HF Space build flake | Multi-stage `openenv-base` Dockerfile + `openenv validate` in CI. |

---

That is the plan. The rest of this commit implements it.
