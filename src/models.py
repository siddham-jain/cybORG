"""
Data models for CybOrg — the multi-agent cybersecurity OpenEnv.

CybOrg exposes one of three role-conditioned cybersecurity tasks (Red, Blue, or
Dual self-play) on a procedurally-generated enterprise network. The action
grammar is text-first / JSON-first so any LLM can play without learning a
custom encoding.

These Pydantic types are the wire format. The Environment expects an action
that successfully validates against ``CybOrgAction`` and emits observations
that conform to ``CybOrgObservation``. Anything that fails Pydantic validation
is treated as a malformed action by the format gate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


# ---------------------------------------------------------------------------
# Action grammar
# ---------------------------------------------------------------------------

# Whitelisted tool names. Reserved OpenEnv names (`reset`, `step`, `state`,
# `close`) deliberately excluded so we never collide with the framework.
RED_TOOLS = (
    "scan",
    "exploit",
    "use_credential",
    "lateral_move",
    "exfiltrate",
    "phish",
    "wait",
)
BLUE_TOOLS = (
    "monitor",
    "analyze_alert",
    "block_ip",
    "isolate_host",
    "patch",
    "rotate_cred",
    "submit_finding",
    "wait",
)
ALL_TOOLS = tuple(set(RED_TOOLS) | set(BLUE_TOOLS))


class CybOrgAction(Action):
    """A single text-style action emitted by the agent.

    The action is intentionally a flat ``(tool, args)`` pair so the grammar is
    easy to express in a system prompt and easy for a verifier to inspect.

    ``role`` is required when the environment is in dual mode and tells us
    which side took this action; in single-role tasks the environment fills it
    in automatically and the field can be omitted.
    """

    tool: str = Field(
        ...,
        description=(
            "Tool name. Red tools: "
            + ", ".join(RED_TOOLS)
            + ". Blue tools: "
            + ", ".join(BLUE_TOOLS)
            + "."
        ),
    )
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool arguments. Each tool defines its own args schema.",
    )
    role: Optional[Literal["red", "blue"]] = Field(
        default=None,
        description=(
            "Acting role. Required only in dual self-play mode; ignored in "
            "single-role tasks where the role is fixed."
        ),
    )


# ---------------------------------------------------------------------------
# Observation grammar
# ---------------------------------------------------------------------------


class CybOrgObservation(Observation):
    """Role-aware, partially observable view of the network.

    The observation is a self-contained dict so an LLM can render it directly
    into a prompt with no extra glue. Fields that the active role cannot see
    are present but ``None`` so the JSON schema stays stable.
    """

    role: Literal["red", "blue"] = Field(..., description="Active role for this turn.")
    task: Literal["red_easy", "blue_easy", "dual"] = Field(
        ..., description="Task mode the environment is currently running."
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        ..., description="World difficulty selected at reset()."
    )
    turn: int = Field(..., ge=0, description="Current turn index (0-based).")

    # World view (red-side fields)
    discovered_hosts: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Hosts the red side has located, with services and ownership.",
    )
    credentials: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Credentials currently held by red.",
    )

    # World view (blue-side fields)
    alerts: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Alerts visible to blue's SOC monitor this episode.",
    )
    service_health: Optional[Dict[str, str]] = Field(
        default=None,
        description="Map of host_id -> 'up'|'isolated'|'down' as seen by blue.",
    )
    green_baseline: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Baseline NPC traffic stats so blue can filter noise.",
    )

    # Shared
    objectives: List[str] = Field(
        default_factory=list, description="Open objectives for the active role."
    )
    objectives_done: List[str] = Field(
        default_factory=list, description="Objectives already satisfied this episode."
    )
    last_action_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Structured result of the last action: ``{ok, stdout, stderr, "
            "events}``. Used by the model to ground its next action."
        ),
    )
    score_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-component contributions to this step's reward, keyed by "
            "rubric column. Useful for trainer dashboards."
        ),
    )
    budget: Dict[str, int] = Field(
        default_factory=dict,
        description="Remaining budgets: steps_left, noise_left.",
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Free-form auxiliary info (e.g. winner string at terminal step, "
            "format error reason if the action was rejected)."
        ),
    )


# ---------------------------------------------------------------------------
# State (server-internal, exposed via /state for debugging)
# ---------------------------------------------------------------------------


class CybOrgState(State):
    """Server-side state. Inherits ``episode_id`` and ``step_count``.

    These extra fields exist mainly so trainers and judges can introspect a
    run without poking at private attributes. They are not part of the
    agent-visible observation.
    """

    task: str = Field(default="red_easy", description="Active task mode.")
    difficulty: str = Field(default="easy", description="Active difficulty tier.")
    seed: Optional[int] = Field(default=None, description="Episode RNG seed.")
    red_score: float = Field(default=0.0, description="Cumulative red reward.")
    blue_score: float = Field(default=0.0, description="Cumulative blue reward.")
    winner: Optional[str] = Field(
        default=None, description="'red'|'blue'|'timeout'|None mid-episode."
    )
    next_role: Literal["red", "blue"] = Field(
        default="red", description="Which role acts on the next step()."
    )
