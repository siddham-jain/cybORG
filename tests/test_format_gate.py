"""Anti-hacking and format-gate tests.

These adversarial cases ensure malformed actions, invalid roles, and forged
finding submissions never produce a positive reward. They are the
regression suite for the reward-hacking guarantees claimed in the README.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cyborg_env.models import CybOrgAction
from cyborg_env.server.cyborg_env_environment import CybOrgEnvironment


def test_action_extra_fields_rejected_by_pydantic():
    with pytest.raises(ValidationError):
        CybOrgAction(tool="scan", args={}, role="red", surprise="hi")


def test_invalid_role_rejected_by_pydantic():
    with pytest.raises(ValidationError):
        CybOrgAction(tool="scan", role="hacker")


def test_blue_tool_in_red_easy_is_format_error():
    env = CybOrgEnvironment()
    env.reset(seed=0, task="red_easy")
    obs = env.step(CybOrgAction(tool="isolate_host", args={"host": "x"}, role="red"))
    assert obs.info.get("format_ok") is False
    assert obs.reward == 0.0


def test_false_finding_does_not_credit_blue():
    env = CybOrgEnvironment()
    env.reset(seed=0, task="blue_easy")
    obs = env.step(
        CybOrgAction(
            tool="submit_finding",
            args={"host": "dmz-00", "vuln": "fake_cve", "actor": "red"},
            role="blue",
        )
    )
    assert (obs.reward or 0) <= 0, "false findings must not be rewarded"


def test_diversity_audit_warns_on_repetition():
    env = CybOrgEnvironment()
    env.reset(seed=0, task="red_easy")
    for _ in range(12):
        env.step(CybOrgAction(tool="wait", args={}, role="red"))
    deltas = env._compute_deltas("red", None, False, None)
    assert deltas["diversity_warn"] is True
