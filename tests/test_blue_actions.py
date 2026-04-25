"""Tests for blue-side action handlers."""

from __future__ import annotations

import random

from cyborg_env.server.sim import EventLog, EventType, WorldGenerator
from cyborg_env.server.sim.blue_actions import (
    blue_isolate_host,
    blue_patch,
    blue_submit_finding,
)


def _setup(seed: int = 1):
    world = WorldGenerator(seed=seed, difficulty="easy").build()
    log = EventLog()
    rng = random.Random(seed)
    return world, log, rng


def test_patch_marks_vuln_patched():
    world, log, rng = _setup()
    host = next(iter(world.hosts.values()))
    vuln = host.vulnerabilities[0]
    out = blue_patch(world, {"host": host.id, "vuln": vuln.id}, rng, log)
    assert out["ok"], out
    assert vuln.patched
    assert log.by_type(EventType.HOST_PATCHED)


def test_isolate_overblock_when_no_red_activity():
    world, log, rng = _setup()
    host = next(iter(world.hosts.values()))
    blue_isolate_host(world, {"host": host.id}, rng, log)
    assert host.isolated
    assert log.by_type(EventType.OVERBLOCK), "isolating a clean host must penalize"


def test_submit_finding_rejected_without_truth():
    world, log, rng = _setup()
    host = next(iter(world.hosts.values()))
    vuln = host.vulnerabilities[0]
    out = blue_submit_finding(
        world, {"host": host.id, "vuln": vuln.id, "actor": "red"}, rng, log
    )
    assert not out["ok"], "no real exploit -> finding rejected"
    assert log.by_type(EventType.FINDING_FALSE)
