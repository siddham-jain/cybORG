"""Event log for CybOrg.

Every consequential thing that happens in the simulator emits an ``Event``.
Reward rubrics, the heuristic blue agent, and the debug observation all read
from this log — never from the agent's free text. That is what makes the
reward objective (and the model's "claim" hard to hack).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    """Closed set of simulator-emitted event types.

    Adding a new event here is a deliberate API change because rubrics key off
    these strings. Do not add events for "agent reasoning" or other free-form
    things — events represent ground-truth state changes only.
    """

    # Red side (attacker)
    HOST_DISCOVERED = "host_discovered"
    EXPLOIT_ATTEMPT = "exploit_attempt"
    EXPLOIT_SUCCESS = "exploit_success"
    EXPLOIT_FAILURE = "exploit_failure"
    INITIAL_ACCESS = "initial_access"
    PRIV_ESC = "priv_esc"
    CRED_OBTAINED = "cred_obtained"
    LATERAL_MOVE = "lateral_move"
    ZONE_CROSSED = "zone_crossed"
    EXFIL_SUCCESS = "exfil_success"
    PHISH_SUCCESS = "phish_success"
    PHISH_FAILURE = "phish_failure"

    # Blue side (defender)
    ALERT_RAISED = "alert_raised"
    ALERT_BENIGN = "alert_benign"  # green NPC produced this
    HOST_ISOLATED = "host_isolated"
    HOST_PATCHED = "host_patched"
    CRED_ROTATED = "cred_rotated"
    IP_BLOCKED = "ip_blocked"
    FINDING_TRUE = "finding_true"
    FINDING_FALSE = "finding_false"
    OVERBLOCK = "overblock"

    # Green NPC
    GREEN_TRAFFIC = "green_traffic"

    # Episode lifecycle
    EPISODE_END = "episode_end"


@dataclass
class Event:
    """A single timestamped event.

    ``data`` carries whatever payload the rubric or UI cares about
    (host_id, vuln_id, asset_id, attribution).
    """

    type: EventType
    turn: int
    role: str  # "red" | "blue" | "green" | "system"
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventLog:
    """Chronologically ordered list of events for a single episode.

    The log doubles as the audit trail: tests assert on it and the README
    demo prints transcripts straight out of it.
    """

    events: List[Event] = field(default_factory=list)

    def emit(self, type_: EventType, turn: int, role: str, **data: Any) -> Event:
        event = Event(type=type_, turn=turn, role=role, data=data)
        self.events.append(event)
        return event

    def by_type(self, type_: EventType) -> List[Event]:
        return [e for e in self.events if e.type == type_]

    def by_turn(self, turn: int) -> List[Event]:
        return [e for e in self.events if e.turn == turn]

    def visible_to_blue(self, since_turn: int = 0) -> List[Event]:
        """Subset of events the blue side actually observes.

        We expose alerts (true and benign) and blue's own activity. Red's
        successes are *not* directly visible — blue must infer them from
        ``ALERT_RAISED`` events, which only fire probabilistically based on
        the vuln's ``detection_rate``.
        """

        visible_types = {
            EventType.ALERT_RAISED,
            EventType.ALERT_BENIGN,
            EventType.HOST_ISOLATED,
            EventType.HOST_PATCHED,
            EventType.CRED_ROTATED,
            EventType.IP_BLOCKED,
            EventType.GREEN_TRAFFIC,
        }
        return [
            e for e in self.events if e.turn >= since_turn and e.type in visible_types
        ]

    def latest(self, type_: Optional[EventType] = None) -> Optional[Event]:
        if type_ is None:
            return self.events[-1] if self.events else None
        for e in reversed(self.events):
            if e.type == type_:
                return e
        return None
