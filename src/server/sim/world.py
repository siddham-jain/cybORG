"""World model for CybOrg.

The world is a directed graph of hosts. Each host belongs to a zone, runs
services, holds data assets, and has a per-role privilege level. Vulnerabilities
have crisp boolean preconditions and outcomes so that a verifier can decide
without ambiguity whether an exploit was legitimate.

This file deliberately keeps the data classes small and JSON-serializable so
they can flow into observations and tests without translation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Leaf entities
# ---------------------------------------------------------------------------


@dataclass
class Vulnerability:
    """A single weakness on a host service.

    Attributes:
        id: Stable identifier exposed to the agent (e.g. ``cve_web_rce``).
        type: ``REMOTE`` (exploitable across the network) or ``LOCAL``
            (requires existing foothold, used for privilege escalation).
        service: The service this vuln lives on (e.g. ``http``, ``ssh``).
        precondition: Free-form key the action handler checks against host
            state (e.g. ``"requires_credential"``); empty string means no
            precondition beyond reachability.
        outcome: One of ``access`` (initial access), ``priv_esc``,
            ``leak_credential``, ``leak_node``. Drives milestone reward.
        success_rate: Probability the exploit lands when attempted.
        detection_rate: Probability the blue side observes the attempt.
        leaks_credential: Optional credential id that an ``leak_credential``
            outcome reveals on success.
        leaks_node: Optional host id that a ``leak_node`` outcome reveals.
    """

    id: str
    type: str
    service: str
    precondition: str = ""
    outcome: str = "access"
    success_rate: float = 0.85
    detection_rate: float = 0.20
    leaks_credential: Optional[str] = None
    leaks_node: Optional[str] = None
    patched: bool = False


@dataclass
class Credential:
    """A reusable login token that a red agent has discovered."""

    id: str
    valid_for_host: str
    valid_for_service: str
    rotated: bool = False


@dataclass
class Asset:
    """A piece of data the red side may exfiltrate.

    ``value`` is on a 1..100 scale and feeds the exfil reward.
    """

    id: str
    value: int
    asset_class: str  # crown_jewel | sensitive | ordinary


@dataclass
class Host:
    """A single machine in the network.

    Many of these fields are world-truth (services, vulnerabilities, assets);
    the partial-observability layer above decides what each role can see.
    """

    id: str
    zone: str
    services: List[str] = field(default_factory=list)
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    assets: List[Asset] = field(default_factory=list)
    firewall_in: Set[str] = field(
        default_factory=set,
        metadata={"doc": "Zones allowed to reach this host (inbound firewall)."},
    )
    firewall_out: Set[str] = field(
        default_factory=set,
        metadata={"doc": "Zones this host can reach (outbound firewall)."},
    )

    # Mutable per-episode state below this line --------------------------
    discovered: bool = False  # red has scanned this host
    owned: bool = False  # red has user-level access
    privileged: bool = False  # red has root/admin
    isolated: bool = False  # blue has cut this host off
    monitored: bool = False  # blue has a sensor on this host

    def vuln(self, vuln_id: str) -> Optional[Vulnerability]:
        """Return a vulnerability by id, or ``None`` if not present."""

        for v in self.vulnerabilities:
            if v.id == vuln_id:
                return v
        return None

    def has_asset(self, asset_id: str) -> bool:
        return any(a.id == asset_id for a in self.assets)


# ---------------------------------------------------------------------------
# Top-level world container
# ---------------------------------------------------------------------------


@dataclass
class GreenNPC:
    """Background user that emits realistic noise.

    Each turn a green NPC may run a short routine (browse the web, send mail,
    print a payslip). These actions show up in the alert stream as benign
    events and force blue to *not* over-block.
    """

    id: str
    role: str  # sales|engineer|finance|it_admin
    home_host: str
    susceptibility: float = 0.1
    awareness: float = 0.5


@dataclass
class World:
    """A fully-instantiated, deterministic episode world.

    Worlds are built once per ``reset()`` by ``WorldGenerator`` and then
    mutated in place as the red/blue actions run. Anything that needs to
    survive a serialization round trip lives on this dataclass.
    """

    seed: int
    difficulty: str
    hosts: Dict[str, Host]
    credentials: Dict[str, Credential] = field(default_factory=dict)
    npcs: Dict[str, GreenNPC] = field(default_factory=dict)
    crown_jewel_asset: Optional[str] = None

    # Red-discovered state (separate so blue cannot read it).
    red_known_hosts: Set[str] = field(default_factory=set)
    red_known_creds: Set[str] = field(default_factory=set)
    exfiltrated_assets: Set[str] = field(default_factory=set)
    pwned_hosts: Set[str] = field(default_factory=set)
    crossed_zones: Set[str] = field(default_factory=set)

    # Red objectives are crisp predicates; we tick them off here.
    red_objectives: List[str] = field(default_factory=list)
    red_objectives_done: Set[str] = field(default_factory=set)

    # Blue findings: which alerts blue has correctly attributed to red.
    blue_findings: Set[str] = field(default_factory=set)
    blue_false_findings: int = 0
    blue_overblocks: int = 0

    # Step accounting.
    step: int = 0
    step_budget: int = 50
    noise_budget: int = 10  # red noise budget
    reference_red_step: int = 0  # cursor into the scripted red trace

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def host(self, host_id: str) -> Optional[Host]:
        return self.hosts.get(host_id)

    def credential(self, cred_id: str) -> Optional[Credential]:
        return self.credentials.get(cred_id)

    def reachable(self, src_zone: str, dst_host: str) -> bool:
        """Return True if a host in ``src_zone`` can reach ``dst_host``."""

        host = self.hosts.get(dst_host)
        if host is None:
            return False
        if host.isolated:
            return False
        return src_zone in host.firewall_in or src_zone == host.zone

    def crown_jewel_owned(self) -> bool:
        """True iff the crown-jewel asset has been exfiltrated by red."""

        if self.crown_jewel_asset is None:
            return False
        return self.crown_jewel_asset in self.exfiltrated_assets
