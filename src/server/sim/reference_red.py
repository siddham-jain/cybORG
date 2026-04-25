"""Scripted reference red attacker.

Used as the built-in opponent when the model is training as blue. The
reference attacker is *intentionally noisy* on easy difficulty so blue gets
plenty of true positives to learn from. As difficulty rises the reference
becomes quieter, increasing blue's burden.

The reference trace is reconstructed greedily from the world's truth: it
scans, exploits, optionally lateralizes, and exfiltrates the crown jewel.
This guarantees admissibility — every world produced by the generator can
in principle be solved.
"""

from __future__ import annotations

import random
from typing import Tuple

from .events import EventLog, EventType
from .world import World


class ReferenceRed:
    """Greedy attacker that walks toward the crown jewel."""

    def __init__(self, rng: random.Random):
        self.rng = rng

    def act(self, world: World, log: EventLog) -> Tuple[str, dict]:
        # Step 1: ensure DMZ ingress is owned.
        dmz_ingress = None
        for host in world.hosts.values():
            if host.zone == "dmz":
                dmz_ingress = host
                break
        if dmz_ingress is None:
            return "wait", {}

        if not dmz_ingress.discovered:
            return "scan", {"target": dmz_ingress.id}

        if not dmz_ingress.owned:
            for vuln in dmz_ingress.vulnerabilities:
                if vuln.type == "REMOTE" and vuln.outcome == "access" and not vuln.patched:
                    return "exploit", {"target": dmz_ingress.id, "vuln": vuln.id}

        # Step 2: pivot through any unowned host we can reach.
        for host in world.hosts.values():
            if host.owned or host.isolated:
                continue
            if not host.discovered:
                if any(
                    owned.zone == host.zone or host.zone in world.host(owned_id).firewall_out
                    for owned_id in world.pwned_hosts
                    for owned in [world.host(owned_id)] if owned is not None
                ):
                    return "scan", {"target": host.id}
            for vuln in host.vulnerabilities:
                if vuln.patched:
                    continue
                if vuln.type == "REMOTE" and host.discovered:
                    return "exploit", {"target": host.id, "vuln": vuln.id}

        # Step 3: exfiltrate any reachable crown jewel.
        if world.crown_jewel_asset is not None:
            for host in world.hosts.values():
                if not host.owned:
                    continue
                if host.has_asset(world.crown_jewel_asset):
                    return "exfiltrate", {
                        "target": host.id,
                        "asset": world.crown_jewel_asset,
                    }

        # Step 4: fall back to wait so we don't burn the noise budget.
        return "wait", {}
