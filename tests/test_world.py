"""Tests for the procedural world generator.

We assert determinism (same seed -> same world) and a couple of structural
invariants (DMZ exists, crown jewel exists, every world has a remote-access
vuln on the entry host so red has *something* to do at t=0).
"""

from __future__ import annotations

from cyborg_env.server.sim import WorldGenerator
from cyborg_env.server.sim.world import Vulnerability


def test_seed_determinism():
    a = WorldGenerator(seed=42, difficulty="easy").build()
    b = WorldGenerator(seed=42, difficulty="easy").build()
    assert sorted(a.hosts) == sorted(b.hosts)
    assert a.crown_jewel_asset == b.crown_jewel_asset


def test_easy_world_size():
    world = WorldGenerator(seed=1, difficulty="easy").build()
    assert len(world.hosts) == 5
    assert any(h.zone == "dmz" for h in world.hosts.values())


def test_entry_host_has_remote_access():
    world = WorldGenerator(seed=1, difficulty="easy").build()
    dmz = next(h for h in world.hosts.values() if h.zone == "dmz")
    assert any(
        v.type == "REMOTE" and v.outcome == "access" and not v.patched
        for v in dmz.vulnerabilities
    ), "entry DMZ host must always have a remote access vuln"


def test_crown_jewel_placed():
    world = WorldGenerator(seed=2, difficulty="medium").build()
    assert world.crown_jewel_asset is not None
    found = any(
        any(a.id == world.crown_jewel_asset for a in host.assets)
        for host in world.hosts.values()
    )
    assert found, "crown jewel asset must live on some host"
