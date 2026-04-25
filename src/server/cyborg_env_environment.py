"""CybOrg Environment.

Wires together:
    - WorldGenerator (procedural worlds keyed by seed + difficulty),
    - Red / Blue action handlers (mutate world + emit events),
    - Green NPC ticker (background noise),
    - Heuristic blue / Reference red (built-in opponents),
    - Red / Blue reward rubrics.

Task modes
----------
``red_easy``    : agent controls Red, heuristic blue runs as opponent.
``blue_easy``   : agent controls Blue, reference red runs as opponent.
``dual``        : agent controls both Red and Blue alternating; useful for
                  self-play and dataset collection.

The mode is set at ``reset()`` via the ``task`` kwarg (or the
``CYBORG_DEFAULT_TASK`` env var) and remains stable for the episode.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import CybOrgAction, CybOrgObservation, CybOrgState, BLUE_TOOLS, RED_TOOLS
except ImportError:  # pragma: no cover - container path
    from models import CybOrgAction, CybOrgObservation, CybOrgState, BLUE_TOOLS, RED_TOOLS

from .rewards import build_blue_rubric, build_red_rubric
from .sim import EventLog, EventType, World, WorldGenerator
from .sim.blue_actions import BLUE_DISPATCH
from .sim.green import step_green
from .sim.heuristic_blue import HeuristicBlue
from .sim.red_actions import RED_DISPATCH
from .sim.reference_red import ReferenceRed


VALID_TASKS = ("red_easy", "blue_easy", "dual")
VALID_DIFFICULTIES = ("easy", "medium", "hard")


class CybOrgEnvironment(Environment):
    """Multi-agent cybersecurity environment for LLM RL.

    The environment is designed so that *both* red and blue can be trained
    against the same simulator without changing any code on the trainer
    side. The reward rubric is selected based on the role of the action
    being evaluated.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        # We swap the rubric in step() based on whose turn it is, so we
        # store both up front.
        self._red_rubric = build_red_rubric()
        self._blue_rubric = build_blue_rubric()
        self.rubric = self._red_rubric  # default; reset() will set the right one
        self._state = CybOrgState(episode_id=str(uuid4()))
        self._world: Optional[World] = None
        self._log: Optional[EventLog] = None
        self._rng: Optional[random.Random] = None
        self._heuristic_blue: Optional[HeuristicBlue] = None
        self._reference_red: Optional[ReferenceRed] = None
        # Counters for delta computation (new_X this step vs total).
        self._prev_red_alerts = 0
        self._prev_findings_true = 0
        self._prev_findings_false = 0
        self._prev_overblocks = 0
        self._prev_pwned = 0
        self._prev_objectives_done = 0
        self._action_fingerprints: List[str] = []
        # Last role to act -> next role to act (for dual mode).
        self._last_acting_role: str = "red"

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CybOrgObservation:
        """Start a fresh episode.

        Optional kwargs:
            task: one of ``red_easy|blue_easy|dual`` (default from env var).
            difficulty: ``easy|medium|hard`` (default ``easy``).
        """

        task = self._normalize_task(kwargs.get("task"))
        difficulty = self._normalize_difficulty(kwargs.get("difficulty"))
        seed = seed if seed is not None else random.randint(0, 2**31 - 1)

        self._rng = random.Random(seed)
        self._world = WorldGenerator(seed=seed, difficulty=difficulty).build()
        self._log = EventLog()
        self._heuristic_blue = HeuristicBlue(rng=random.Random(seed + 1))
        self._reference_red = ReferenceRed(rng=random.Random(seed + 2))

        self._state = CybOrgState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task=task,
            difficulty=difficulty,
            seed=seed,
            next_role="red" if task != "blue_easy" else "blue",
        )
        # Reset per-episode counters.
        self._prev_red_alerts = 0
        self._prev_findings_true = 0
        self._prev_findings_false = 0
        self._prev_overblocks = 0
        self._prev_pwned = 0
        self._prev_objectives_done = 0
        self._action_fingerprints = []
        self._last_acting_role = self._state.next_role

        # Choose the rubric matching the very first acting role.
        self.rubric = self._rubric_for(self._state.next_role)
        self._reset_rubric()

        return self._make_observation(
            role=self._state.next_role,
            last_result=None,
            done=False,
            info={"format_ok": True, "message": "episode_started"},
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(
        self,
        action: CybOrgAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CybOrgObservation:
        if self._world is None or self._log is None:
            return self._error_observation("call reset() before step()")

        # Determine acting role.
        role = self._role_for_action(action)

        # Validate tool against role-allowed set; treat as malformed if not.
        allowed = RED_TOOLS if role == "red" else BLUE_TOOLS
        if action.tool not in allowed:
            self._world.step += 1
            self._state.step_count += 1
            obs = self._make_observation(
                role=self._next_role_after(role),
                last_result={"ok": False, "stderr": f"tool {action.tool!r} not allowed for role {role}"},
                done=False,
                info={
                    "format_ok": False,
                    "format_error": f"tool {action.tool!r} not allowed for {role}",
                },
            )
            # Format gate forces reward=0; set it explicitly so callers always
            # see a numeric reward (rather than the Pydantic default ``None``).
            obs.reward = 0.0
            return obs

        # Execute the action via the dispatch table.
        dispatch = RED_DISPATCH if role == "red" else BLUE_DISPATCH
        handler = dispatch[action.tool]
        last_result = handler(self._world, dict(action.args), self._rng, self._log)

        # Track fingerprints for the diversity audit.
        self._action_fingerprints.append(self._fingerprint(role, action))

        # Run the opposing scripted policy if in single-role mode.
        opp_result = self._run_opponent(role)

        # Tick green NPCs once per "world step" (a red+blue exchange).
        step_green(self._world, self._rng, self._log)

        self._world.step += 1
        self._state.step_count += 1

        done, winner, reason = self._check_episode_end(role)

        # Compute deltas for the rubric.
        deltas = self._compute_deltas(role, action, done, winner)

        info: Dict[str, Any] = {
            "format_ok": True,
            "winner": winner,
            "reason": reason,
            "opponent_action": opp_result,
            f"{role}_deltas": deltas,
        }
        # Always write both delta columns so dashboards can pick either.
        if role == "red":
            info.setdefault("blue_deltas", self._compute_deltas("blue", None, done, winner))
        else:
            info.setdefault("red_deltas", self._compute_deltas("red", None, done, winner))

        # Pick the rubric matching the acting role and let it score.
        self.rubric = self._rubric_for(role)

        next_role = self._next_role_after(role)
        observation = self._make_observation(
            role=next_role, last_result=last_result, done=done, info=info,
        )
        observation.reward = float(self.rubric(action, observation))

        # Track running scores on state.
        if role == "red":
            self._state.red_score += observation.reward or 0.0
        else:
            self._state.blue_score += observation.reward or 0.0
        self._state.winner = winner
        self._state.next_role = next_role
        self._last_acting_role = role
        return observation

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def state(self) -> CybOrgState:
        return self._state

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata

        return EnvironmentMetadata(
            name="CybOrg",
            description=(
                "Multi-agent cybersecurity RL environment for LLMs. "
                "Trains attackers, defenders, and the noisy users between them."
            ),
            version="0.1.0",
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _normalize_task(self, task: Optional[str]) -> str:
        task = task or os.environ.get("CYBORG_DEFAULT_TASK", "red_easy")
        if task not in VALID_TASKS:
            return "red_easy"
        return task

    def _normalize_difficulty(self, difficulty: Optional[str]) -> str:
        difficulty = difficulty or os.environ.get("CYBORG_DEFAULT_DIFFICULTY", "easy")
        if difficulty not in VALID_DIFFICULTIES:
            return "easy"
        return difficulty

    def _rubric_for(self, role: str):
        return self._red_rubric if role == "red" else self._blue_rubric

    def _role_for_action(self, action: CybOrgAction) -> str:
        if self._state.task == "red_easy":
            return "red"
        if self._state.task == "blue_easy":
            return "blue"
        # dual mode: respect explicit role if present, else alternate.
        if action.role in {"red", "blue"}:
            return action.role  # type: ignore[return-value]
        return self._state.next_role

    def _next_role_after(self, role: str) -> str:
        if self._state.task == "red_easy":
            return "red"
        if self._state.task == "blue_easy":
            return "blue"
        return "blue" if role == "red" else "red"

    def _run_opponent(self, played_role: str) -> Optional[Dict[str, Any]]:
        """Run the scripted opposing role for this turn (if any)."""

        if self._state.task == "dual":
            return None
        if played_role == "red":
            tool, args = self._heuristic_blue.act(self._world, self._log)
            handler = BLUE_DISPATCH[tool]
            result = handler(self._world, args, self._rng, self._log)
            return {"role": "blue", "tool": tool, "args": args, "result": result}
        if played_role == "blue":
            tool, args = self._reference_red.act(self._world, self._log)
            handler = RED_DISPATCH[tool]
            result = handler(self._world, args, self._rng, self._log)
            return {"role": "red", "tool": tool, "args": args, "result": result}
        return None

    def _check_episode_end(self, played_role: str) -> tuple[bool, Optional[str], Optional[str]]:
        if self._world is None:
            return True, None, "no_world"
        if self._world.crown_jewel_owned():
            return True, "red", "crown_jewel_exfiltrated"
        # Blue wins if the crown jewel host is patched-and-isolated AND red's
        # paths have been broken: simplest victory condition is step budget
        # exhausted with crown jewel still safe.
        if self._world.step >= self._world.step_budget:
            return True, "blue", "step_budget_exhausted"
        return False, None, None

    def _compute_deltas(
        self, role: str, action: Optional[CybOrgAction], done: bool, winner: Optional[str]
    ) -> Dict[str, Any]:
        """Compute per-step reward deltas for the rubric, role-conditioned."""

        if role == "red":
            new_pwned = max(0, len(self._world.pwned_hosts) - self._prev_pwned)
            self._prev_pwned = len(self._world.pwned_hosts)
            new_objectives = max(
                0, len(self._world.red_objectives_done) - self._prev_objectives_done
            )
            self._prev_objectives_done = len(self._world.red_objectives_done)
            # Sum value of *new* exfils this turn.
            exfil_value_share = self._exfil_value_this_turn() / 100.0
            # New red-attributed alerts this turn:
            current_red_alerts = sum(
                1 for e in self._log.events
                if e.type == EventType.ALERT_RAISED and e.data.get("attribution") == "red"
            )
            new_red_alerts = max(0, current_red_alerts - self._prev_red_alerts)
            self._prev_red_alerts = current_red_alerts
            return {
                "new_pwned": new_pwned,
                "new_objectives": new_objectives,
                "exfil_value_share": exfil_value_share,
                "zero_new_alerts": new_red_alerts == 0,
                "new_red_alerts": new_red_alerts,
                "fake_claim": False,  # explicit hook; reserved for claim_objective
                "done": done,
                "winner": winner,
                "diversity_warn": self._diversity_warn(),
            }
        # Blue
        current_findings_true = self._world.blue_findings.__len__()
        current_findings_false = self._world.blue_false_findings
        current_overblocks = self._world.blue_overblocks
        new_findings_true = max(0, current_findings_true - self._prev_findings_true)
        new_findings_false = max(0, current_findings_false - self._prev_findings_false)
        new_overblocks = max(0, current_overblocks - self._prev_overblocks)
        self._prev_findings_true = current_findings_true
        self._prev_findings_false = current_findings_false
        self._prev_overblocks = current_overblocks
        return {
            "new_findings_true": new_findings_true,
            "new_findings_false": new_findings_false,
            "new_containments": self._new_containments_this_turn(),
            "new_overblocks": new_overblocks,
            "continuity_drop": self._continuity_drop(),
            "done": done,
            "winner": winner,
        }

    def _exfil_value_this_turn(self) -> int:
        total = 0
        for e in self._log.by_turn(self._world.step):
            if e.type == EventType.EXFIL_SUCCESS:
                asset_id = e.data.get("asset")
                for host in self._world.hosts.values():
                    for a in host.assets:
                        if a.id == asset_id:
                            total += a.value
        return total

    def _new_containments_this_turn(self) -> int:
        return sum(
            1
            for e in self._log.by_turn(self._world.step)
            if e.type in {EventType.HOST_ISOLATED, EventType.HOST_PATCHED, EventType.CRED_ROTATED}
        )

    def _continuity_drop(self) -> float:
        if not self._world.hosts:
            return 0.0
        isolated = sum(1 for h in self._world.hosts.values() if h.isolated)
        return isolated / len(self._world.hosts)

    def _diversity_warn(self) -> bool:
        if len(self._action_fingerprints) < 6:
            return False
        recent = self._action_fingerprints[-10:]
        most_common = max(set(recent), key=recent.count)
        return recent.count(most_common) / len(recent) > 0.7

    def _fingerprint(self, role: str, action: CybOrgAction) -> str:
        return f"{role}:{action.tool}:{sorted(action.args.keys())}"

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------
    def _make_observation(
        self,
        role: str,
        last_result: Optional[Dict[str, Any]],
        done: bool,
        info: Dict[str, Any],
    ) -> CybOrgObservation:
        world = self._world
        if world is None:
            return CybOrgObservation(
                role="red",
                task="red_easy",
                difficulty="easy",
                turn=0,
                done=True,
                reward=0.0,
                info={"error": "world not initialized"},
            )

        if role == "red":
            discovered = [
                {
                    "id": h.id,
                    "zone": h.zone,
                    "services": h.services,
                    "owned": h.owned,
                    "privileged": h.privileged,
                    "vulnerabilities": [
                        v.id for v in h.vulnerabilities if not v.patched
                    ],
                    "isolated": h.isolated,
                }
                for h in world.hosts.values()
                if h.discovered
            ]
            credentials = [
                {
                    "id": c.id,
                    "host": c.valid_for_host,
                    "service": c.valid_for_service,
                    "rotated": c.rotated,
                }
                for c in world.credentials.values()
                if c.id in world.red_known_creds
            ]
            obs_kwargs = dict(
                discovered_hosts=discovered,
                credentials=credentials,
                objectives=[o for o in world.red_objectives if o not in world.red_objectives_done],
                objectives_done=list(world.red_objectives_done),
            )
        else:  # blue
            visible = self._log.visible_to_blue() if self._log else []
            alerts = [
                {
                    "id": idx,
                    "turn": e.turn,
                    "kind": e.data.get("kind"),
                    "host": e.data.get("host"),
                    "type": e.type.value,
                }
                for idx, e in enumerate(self._log.events) if self._log
                if e.type in {EventType.ALERT_RAISED, EventType.ALERT_BENIGN}
            ]
            service_health = {
                h.id: ("isolated" if h.isolated else "up") for h in world.hosts.values()
            }
            green_baseline = {
                "npc_count": len(world.npcs),
                "benign_alerts_seen": sum(
                    1 for e in (self._log.events if self._log else []) if e.type == EventType.ALERT_BENIGN
                ),
            }
            obs_kwargs = dict(
                alerts=alerts,
                service_health=service_health,
                green_baseline=green_baseline,
                objectives=["prevent crown_jewel exfiltration"],
                objectives_done=[],
            )

        observation = CybOrgObservation(
            role=role,  # type: ignore[arg-type]
            task=self._state.task,  # type: ignore[arg-type]
            difficulty=self._state.difficulty,  # type: ignore[arg-type]
            turn=world.step,
            last_action_result=last_result,
            score_breakdown={},
            budget={
                "steps_left": max(0, world.step_budget - world.step),
                "noise_left": max(0, world.noise_budget - self._prev_red_alerts),
            },
            info={**info, "events_total": len(self._log.events) if self._log else 0},
            done=done,
            **obs_kwargs,
        )
        return observation

    def _error_observation(self, message: str) -> CybOrgObservation:
        return CybOrgObservation(
            role="red",
            task="red_easy",
            difficulty="easy",
            turn=0,
            done=True,
            reward=0.0,
            info={"error": message, "format_ok": False},
        )
