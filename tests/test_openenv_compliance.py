"""Tests that verify the OpenEnv contract is satisfied.

These checks mirror the compliance checklist in PLAN.md so a regression on
the framework integration shows up immediately. They do NOT require a
running HTTP server — we exercise the in-process Environment subclass.
"""

from __future__ import annotations

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from cyborg_env import CybOrg, CybOrgAction, CybOrgObservation, CybOrgState
from cyborg_env.server.cyborg_env_environment import CybOrgEnvironment


def test_action_observation_types_are_openenv_subclasses():
    assert issubclass(CybOrgAction, Action)
    assert issubclass(CybOrgObservation, Observation)
    assert issubclass(CybOrgState, State)


def test_environment_inherits_from_openenv_base():
    assert issubclass(CybOrgEnvironment, Environment)


def test_environment_has_required_methods():
    env = CybOrgEnvironment()
    assert callable(getattr(env, "reset"))
    assert callable(getattr(env, "step"))
    assert isinstance(getattr(type(env), "state").fget(env), State)


def test_no_reserved_tool_names_in_action_grammar():
    """``reset`` / ``step`` / ``state`` / ``close`` must not appear as tools."""

    from cyborg_env.models import RED_TOOLS, BLUE_TOOLS

    reserved = {"reset", "step", "state", "close"}
    assert reserved.isdisjoint(RED_TOOLS)
    assert reserved.isdisjoint(BLUE_TOOLS)


def test_observation_contains_done_and_reward_fields():
    obs = CybOrgObservation(role="red", task="red_easy", difficulty="easy", turn=0)
    assert hasattr(obs, "done")
    assert hasattr(obs, "reward")


def test_reset_accepts_seed_and_episode_id():
    env = CybOrgEnvironment()
    obs = env.reset(seed=123, episode_id="ep-001")
    assert env.state.episode_id == "ep-001"
    assert env.state.seed == 123
    assert obs.role in {"red", "blue"}


def test_client_subclasses_envclient():
    from openenv.core import EnvClient

    assert issubclass(CybOrg, EnvClient)
