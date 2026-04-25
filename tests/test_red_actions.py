"""Tests for red-side action handlers."""

from __future__ import annotations

import random

from cyborg_env.server.sim import EventLog, EventType, WorldGenerator
from cyborg_env.server.sim.red_actions import (
    red_exploit,
    red_exfiltrate,
    red_scan,
)


def _setup(seed: int = 1, difficulty: str = "easy"):
    world = WorldGenerator(seed=seed, difficulty=difficulty).build()
    log = EventLog()
    rng = random.Random(seed + 100)
    return world, log, rng


def test_scan_emits_host_discovered():
    world, log, rng = _setup()
    dmz = next(h for h in world.hosts.values() if h.zone == "dmz")
    result = red_scan(world, {"target": dmz.id}, rng, log)
    assert result["ok"], result
    assert dmz.id in world.red_known_hosts
    assert log.by_type(EventType.HOST_DISCOVERED), "must emit HOST_DISCOVERED"


def test_exploit_requires_scan_first():
    world, log, rng = _setup()
    dmz = next(h for h in world.hosts.values() if h.zone == "dmz")
    vuln = next(v for v in dmz.vulnerabilities if v.type == "REMOTE")
    out = red_exploit(world, {"target": dmz.id, "vuln": vuln.id}, rng, log)
    assert not out["ok"], "exploit before scan must fail"


def test_exploit_initial_access_path():
    """Brute the rng so we hit a deterministic success."""
    world, log, _ = _setup(seed=42)
    dmz = next(h for h in world.hosts.values() if h.zone == "dmz")
    vuln = next(
        v for v in dmz.vulnerabilities if v.type == "REMOTE" and v.outcome == "access"
    )
    # Force success by mutating success_rate to 1.0 and detection to 0.
    vuln.success_rate = 1.0
    vuln.detection_rate = 0.0
    rng = random.Random(0)
    red_scan(world, {"target": dmz.id}, rng, log)
    out = red_exploit(world, {"target": dmz.id, "vuln": vuln.id}, rng, log)
    assert out["ok"], out
    assert dmz.owned
    assert dmz.id in world.pwned_hosts


def test_exfiltrate_requires_owned_host():
    world, log, rng = _setup(seed=11)
    crown_host = next(
        h for h in world.hosts.values() if any(a.asset_class == "crown_jewel" for a in h.assets)
    )
    crown_asset = next(a for a in crown_host.assets if a.asset_class == "crown_jewel")
    out = red_exfiltrate(
        world, {"target": crown_host.id, "asset": crown_asset.id}, rng, log
    )
    assert not out["ok"], "cannot exfiltrate before owning the host"
