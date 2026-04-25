"""Tests for reward rubrics and the format gate."""

from __future__ import annotations

from cyborg_env.models import CybOrgAction, CybOrgObservation
from cyborg_env.server.rewards import (
    FormatGateRubric,
    build_blue_rubric,
    build_red_rubric,
)


def _obs(**info) -> CybOrgObservation:
    return CybOrgObservation(
        role="red", task="red_easy", difficulty="easy", turn=0, info=info
    )


def test_format_gate_blocks_malformed_action():
    rubric = build_red_rubric()
    obs = _obs(format_ok=False, red_deltas={"new_pwned": 5, "done": False})
    reward = rubric(CybOrgAction(tool="scan", role="red"), obs)
    assert reward == 0.0


def test_red_rubric_credits_new_pwned_host():
    rubric = build_red_rubric()
    obs = _obs(
        format_ok=True,
        red_deltas={"new_pwned": 1, "new_objectives": 1, "done": False},
    )
    reward = rubric(CybOrgAction(tool="exploit", role="red"), obs)
    # +0.10 (host) + 0.20 (objective) - 0.01 (step cost) = 0.29
    assert abs(reward - 0.29) < 1e-6


def test_blue_rubric_credits_finding_and_penalizes_overblock():
    rubric = build_blue_rubric()
    obs = CybOrgObservation(
        role="blue",
        task="blue_easy",
        difficulty="easy",
        turn=0,
        info={
            "format_ok": True,
            "blue_deltas": {
                "new_findings_true": 1,
                "new_overblocks": 1,
                "done": False,
            },
        },
    )
    reward = rubric(CybOrgAction(tool="submit_finding", role="blue"), obs)
    # 0.15 - 0.15 - 0.01 = -0.01
    assert abs(reward - (-0.01)) < 1e-6


def test_terminal_red_win_bonus():
    rubric = build_red_rubric()
    obs = _obs(
        format_ok=True,
        red_deltas={"done": True, "winner": "red"},
    )
    reward = rubric(CybOrgAction(tool="exfiltrate", role="red"), obs)
    assert reward >= 0.95  # +1 terminal less per-step cost (-0.01)
