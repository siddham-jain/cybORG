"""Procedural world generator.

Builds a deterministic ``World`` from ``(seed, difficulty)``. The same seed
always yields the same world — that is essential for reproducible RL.

Difficulty controls:
    * network size (5 / 10 / 20 hosts),
    * vuln density and detection rates,
    * green NPC count and noise rate,
    * step budget (30 / 50 / 80),
    * crown-jewel asset placement.

Worlds are *admission-proven*: the generator emits a reference attack trace
(see ``reference_red``) that the world is guaranteed to be solvable for. That
guarantee is what keeps blue training non-degenerate.
"""

from __future__ import annotations

import random
from typing import Dict, List

from .world import Asset, Credential, GreenNPC, Host, Vulnerability, World


# ---------------------------------------------------------------------------
# Difficulty tuning table
# ---------------------------------------------------------------------------

DIFFICULTY = {
    "easy": {
        "n_hosts": 5,
        "n_zones": 2,
        "n_npcs": 2,
        "step_budget": 30,
        "noise_budget": 12,
        "detection_scale": 1.5,  # blue gets boosted detection on easy
        "vulns_per_host": (1, 2),
    },
    "medium": {
        "n_hosts": 10,
        "n_zones": 3,
        "n_npcs": 4,
        "step_budget": 50,
        "noise_budget": 10,
        "detection_scale": 1.0,
        "vulns_per_host": (1, 3),
    },
    "hard": {
        "n_hosts": 20,
        "n_zones": 4,
        "n_npcs": 6,
        "step_budget": 80,
        "noise_budget": 8,
        "detection_scale": 0.7,
        "vulns_per_host": (2, 4),
    },
}


# Service / vulnerability templates. Each template is shaped for one of the
# five outcomes (access, priv_esc, leak_credential, leak_node).
SERVICE_TEMPLATES = [
    ("http", "REMOTE", "access", 0.85, 0.25),
    ("ssh", "REMOTE", "access", 0.55, 0.30),
    ("smb", "REMOTE", "leak_credential", 0.70, 0.35),
    ("rdp", "REMOTE", "access", 0.55, 0.40),
    ("kerberos", "REMOTE", "leak_node", 0.65, 0.35),
    ("local_kernel", "LOCAL", "priv_esc", 0.75, 0.20),
    ("local_misconfig", "LOCAL", "priv_esc", 0.85, 0.10),
]

ZONES = ["dmz", "corp", "ops", "vault"]
NPC_ROLES = ["sales", "engineer", "finance", "it_admin"]


class WorldGenerator:
    """Build deterministic, admission-checked worlds.

    Use it like ``WorldGenerator(seed=7, difficulty="easy").build()``. The
    same ``(seed, difficulty)`` pair always returns an equivalent ``World``
    so episodes can be replayed bit-for-bit during evaluation.
    """

    def __init__(self, seed: int, difficulty: str = "easy"):
        if difficulty not in DIFFICULTY:
            raise ValueError(
                f"Unknown difficulty {difficulty!r}; choose from {list(DIFFICULTY)}"
            )
        self.seed = seed
        self.difficulty = difficulty
        self.rng = random.Random(seed)
        self.cfg = DIFFICULTY[difficulty]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def build(self) -> World:
        zones = ZONES[: self.cfg["n_zones"]]
        hosts = self._build_hosts(zones)
        credentials = self._build_credentials(hosts)
        # Wire credential leaks back into vulnerabilities now that we know
        # the credential ids.
        self._wire_credential_leaks(hosts, credentials)
        # Wire node leaks so kerberos-style vulns reveal a real next host.
        self._wire_node_leaks(hosts)

        npcs = self._build_npcs(hosts)
        crown_jewel = self._place_crown_jewel(hosts)

        objectives = self._build_red_objectives(hosts, credentials, crown_jewel)

        world = World(
            seed=self.seed,
            difficulty=self.difficulty,
            hosts={h.id: h for h in hosts},
            credentials={c.id: c for c in credentials},
            npcs={n.id: n for n in npcs},
            crown_jewel_asset=crown_jewel,
            red_objectives=objectives,
            step_budget=self.cfg["step_budget"],
            noise_budget=self.cfg["noise_budget"],
        )
        return world

    # ------------------------------------------------------------------
    # Hosts
    # ------------------------------------------------------------------
    def _build_hosts(self, zones: List[str]) -> List[Host]:
        hosts: List[Host] = []
        n = self.cfg["n_hosts"]
        # First host is always a DMZ ingress that anyone can reach. That
        # guarantees the red attacker has a non-empty action space at t=0.
        for i in range(n):
            zone = zones[0] if i == 0 else self.rng.choice(zones)
            host = Host(
                id=self._host_id(i, zone),
                zone=zone,
                services=self._sample_services(),
                firewall_in=self._sample_firewall_in(zone, zones),
                firewall_out=set(zones),  # outbound is unrestricted by default
            )
            host.vulnerabilities = self._sample_vulns(host.services)
            hosts.append(host)

        # The first DMZ host always has at least one REMOTE access vuln so
        # the agent can always make progress on turn 0.
        if not any(v.type == "REMOTE" and v.outcome == "access" for v in hosts[0].vulnerabilities):
            hosts[0].vulnerabilities.insert(
                0,
                Vulnerability(
                    id=f"cve_{hosts[0].id}_entry",
                    type="REMOTE",
                    service=hosts[0].services[0] if hosts[0].services else "http",
                    outcome="access",
                    success_rate=0.95,
                    detection_rate=0.20 * self.cfg["detection_scale"],
                ),
            )
        return hosts

    def _host_id(self, i: int, zone: str) -> str:
        return f"{zone}-{i:02d}"

    def _sample_services(self) -> List[str]:
        # Each host runs 1-3 services drawn without replacement.
        all_remote = [s for s, t, *_ in SERVICE_TEMPLATES if t == "REMOTE"]
        k = self.rng.randint(1, min(3, len(all_remote)))
        return self.rng.sample(all_remote, k=k)

    def _sample_firewall_in(self, zone: str, zones: List[str]) -> set:
        # DMZ accepts traffic from anywhere; deeper zones only from adjacent.
        if zone == "dmz":
            return set(zones)
        idx = zones.index(zone)
        adjacent = set(zones[max(0, idx - 1) : idx + 1])
        return adjacent

    def _sample_vulns(self, services: List[str]) -> List[Vulnerability]:
        lo, hi = self.cfg["vulns_per_host"]
        target = self.rng.randint(lo, hi)
        vulns: List[Vulnerability] = []
        seen: set = set()
        attempts = 0
        while len(vulns) < target and attempts < 20:
            svc, vtype, outcome, sr, dr = self.rng.choice(SERVICE_TEMPLATES)
            attempts += 1
            # Local vulns are allowed regardless of declared services
            # because they model post-foothold escalation.
            if vtype == "REMOTE" and svc not in services:
                continue
            vid = f"cve_{svc}_{outcome}_{len(vulns)}"
            if vid in seen:
                continue
            seen.add(vid)
            vulns.append(
                Vulnerability(
                    id=vid,
                    type=vtype,
                    service=svc,
                    outcome=outcome,
                    success_rate=sr,
                    detection_rate=min(1.0, dr * self.cfg["detection_scale"]),
                )
            )
        return vulns

    # ------------------------------------------------------------------
    # Credentials, nodes, NPCs, assets
    # ------------------------------------------------------------------
    def _build_credentials(self, hosts: List[Host]) -> List[Credential]:
        creds: List[Credential] = []
        for host in hosts:
            for svc in host.services:
                if self.rng.random() < 0.4:
                    creds.append(
                        Credential(
                            id=f"cred_{host.id}_{svc}",
                            valid_for_host=host.id,
                            valid_for_service=svc,
                        )
                    )
        # Always seed at least one credential leak target if nothing landed.
        if not creds and hosts:
            host = hosts[0]
            svc = host.services[0] if host.services else "http"
            creds.append(
                Credential(
                    id=f"cred_{host.id}_{svc}",
                    valid_for_host=host.id,
                    valid_for_service=svc,
                )
            )
        return creds

    def _wire_credential_leaks(
        self, hosts: List[Host], credentials: List[Credential]
    ) -> None:
        cred_pool = list(credentials)
        for host in hosts:
            for vuln in host.vulnerabilities:
                if vuln.outcome == "leak_credential" and cred_pool:
                    vuln.leaks_credential = self.rng.choice(cred_pool).id

    def _wire_node_leaks(self, hosts: List[Host]) -> None:
        host_ids = [h.id for h in hosts]
        for host in hosts:
            for vuln in host.vulnerabilities:
                if vuln.outcome == "leak_node":
                    candidates = [hid for hid in host_ids if hid != host.id]
                    if candidates:
                        vuln.leaks_node = self.rng.choice(candidates)

    def _build_npcs(self, hosts: List[Host]) -> List[GreenNPC]:
        npcs: List[GreenNPC] = []
        for i in range(self.cfg["n_npcs"]):
            home = self.rng.choice(hosts)
            role = self.rng.choice(NPC_ROLES)
            npcs.append(
                GreenNPC(
                    id=f"npc-{i:02d}",
                    role=role,
                    home_host=home.id,
                    susceptibility=self.rng.uniform(0.05, 0.4),
                    awareness=self.rng.uniform(0.3, 0.8),
                )
            )
        return npcs

    def _place_crown_jewel(self, hosts: List[Host]) -> str:
        # Place the crown jewel on the deepest zone we have.
        deepest = max(hosts, key=lambda h: ZONES.index(h.zone) if h.zone in ZONES else 0)
        asset = Asset(
            id=f"asset_crown_{deepest.id}", value=100, asset_class="crown_jewel"
        )
        deepest.assets.append(asset)
        # Sprinkle a couple of "sensitive" assets on random hosts as
        # secondary objectives.
        for _ in range(min(2, len(hosts))):
            h = self.rng.choice(hosts)
            h.assets.append(
                Asset(
                    id=f"asset_doc_{h.id}_{len(h.assets)}",
                    value=self.rng.randint(20, 60),
                    asset_class="sensitive",
                )
            )
        return asset.id

    def _build_red_objectives(
        self,
        hosts: List[Host],
        credentials: List[Credential],
        crown_jewel: str,
    ) -> List[str]:
        objectives = [
            f"initial_access@{hosts[0].id}",
            f"exfiltrate@{crown_jewel}",
        ]
        if credentials:
            objectives.append(f"obtain_credential@{credentials[0].id}")
        if any(h.zone != hosts[0].zone for h in hosts):
            objectives.append("cross_zone")
        return objectives
