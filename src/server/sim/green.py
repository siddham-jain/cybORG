"""Green NPC traffic generator.

Green NPCs produce benign-but-noisy events each turn. The point is to give
blue a *non-trivial* alert stream to filter — without green noise, blue
could just submit_finding on every alert and trivially win.
"""

from __future__ import annotations

import random

from .events import EventLog, EventType
from .world import World


def step_green(world: World, rng: random.Random, log: EventLog) -> None:
    """Tick all NPCs. Each NPC has a chance to emit a benign alert.

    The benign alert format mirrors a real ALERT but the event type is
    ALERT_BENIGN so analyze_alert can distinguish it.
    """

    for npc in world.npcs.values():
        if rng.random() < 0.3:
            host = world.host(npc.home_host)
            if host is None or host.isolated:
                continue
            log.emit(
                EventType.GREEN_TRAFFIC,
                world.step,
                "green",
                npc=npc.id,
                host=host.id,
                npc_role=npc.role,
            )
            if rng.random() < 0.4:
                log.emit(
                    EventType.ALERT_BENIGN,
                    world.step,
                    "green",
                    npc=npc.id,
                    host=host.id,
                    kind="user_activity",
                    attribution="green",
                )
