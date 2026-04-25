"""Scripted heuristic blue agent.

Used as the built-in opponent when the model is training as red. The
heuristic is intentionally noisy (it sometimes over-blocks, sometimes
ignores real alerts) so red has space to learn but is never unopposed.
"""

from __future__ import annotations

import random
from typing import Tuple

from .blue_actions import BLUE_DISPATCH
from .events import EventLog, EventType
from .world import World


class HeuristicBlue:
    """Stateful policy that returns one ``(tool, args)`` per turn.

    The strategy is:
    1. Maintain monitors on a small rotating subset of hosts.
    2. Inspect any new alert; submit a finding only when the analyzed alert
       points to a true exploit.
    3. Patch a known-vulnerable host probabilistically.
    4. Isolate a host once it has accumulated >= 2 alerts.
    """

    def __init__(self, rng: random.Random):
        self.rng = rng
        self._patched_keys: set = set()

    def act(self, world: World, log: EventLog) -> Tuple[str, dict]:
        # 1) Submit findings for any unconfirmed exploit-success events.
        for ev in log.events:
            if ev.type != EventType.EXPLOIT_SUCCESS:
                continue
            host = ev.data.get("host")
            vuln = ev.data.get("vuln")
            key = f"{host}:{vuln}"
            if key in world.blue_findings:
                continue
            if self.rng.random() < 0.6:
                return "submit_finding", {"host": host, "vuln": vuln, "actor": "red"}

        # 2) Isolate a host that has >= 2 alerts and isn't already isolated.
        host_alerts: dict = {}
        for ev in log.by_type(EventType.ALERT_RAISED):
            host_alerts[ev.data.get("host")] = host_alerts.get(ev.data.get("host"), 0) + 1
        for hid, count in host_alerts.items():
            host = world.host(hid)
            if host is None or host.isolated:
                continue
            if count >= 2 and self.rng.random() < 0.5:
                return "isolate_host", {"host": hid}

        # 3) Patch something.
        if self.rng.random() < 0.3:
            for host in world.hosts.values():
                if host.isolated:
                    continue
                for vuln in host.vulnerabilities:
                    if vuln.patched:
                        continue
                    key = f"{host.id}:{vuln.id}"
                    if key in self._patched_keys:
                        continue
                    self._patched_keys.add(key)
                    return "patch", {"host": host.id, "vuln": vuln.id}

        # 4) Rotate monitors.
        targets = [h for h in world.hosts.values() if not h.monitored and not h.isolated]
        if targets:
            host = self.rng.choice(targets)
            return "monitor", {"host": host.id}

        return "wait", {}
