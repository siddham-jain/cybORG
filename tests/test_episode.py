"""End-to-end episode tests against the in-process Environment."""

from __future__ import annotations

from cyborg_env.models import CybOrgAction
from cyborg_env.server.cyborg_env_environment import CybOrgEnvironment


def test_reset_returns_red_observation_in_red_easy():
    env = CybOrgEnvironment()
    obs = env.reset(seed=7, task="red_easy", difficulty="easy")
    assert obs.role == "red"
    assert obs.task == "red_easy"
    assert obs.difficulty == "easy"
    assert obs.discovered_hosts is not None  # red gets the network view
    assert obs.alerts is None  # red does NOT see the blue alert stream


def test_reset_returns_blue_observation_in_blue_easy():
    env = CybOrgEnvironment()
    obs = env.reset(seed=7, task="blue_easy", difficulty="easy")
    assert obs.role == "blue"
    assert obs.alerts is not None
    assert obs.discovered_hosts is None


def test_step_executes_scan_and_increments_turn():
    env = CybOrgEnvironment()
    obs = env.reset(seed=11, task="red_easy")
    # Find any host in DMZ from full list (we have to peek into state for this).
    dmz_id = next(
        h["id"] for h in (obs.discovered_hosts or []) + _all_hosts(env) if h.get("zone") == "dmz"
    )
    obs2 = env.step(CybOrgAction(tool="scan", args={"target": dmz_id}, role="red"))
    assert obs2.turn == obs.turn + 1
    assert obs2.last_action_result is not None


def test_invalid_tool_marks_format_error():
    env = CybOrgEnvironment()
    env.reset(seed=11, task="red_easy")
    out = env.step(CybOrgAction(tool="hack_everything", args={}, role="red"))
    assert out.info.get("format_ok") is False
    assert out.reward == 0.0


def test_episode_terminates_on_step_budget():
    env = CybOrgEnvironment()
    env.reset(seed=1, task="red_easy", difficulty="easy")
    obs = None
    for _ in range(50):
        obs = env.step(CybOrgAction(tool="wait", args={}, role="red"))
        if obs.done:
            break
    assert obs is not None and obs.done


def _all_hosts(env: CybOrgEnvironment):
    """Test helper: fetch all hosts straight from the simulator."""

    if env._world is None:
        return []
    return [
        {"id": h.id, "zone": h.zone, "owned": h.owned}
        for h in env._world.hosts.values()
    ]
