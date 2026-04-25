"""CybOrg simulator package.

The simulator is the source of truth for the world. Reward rubrics read
events from here; they never parse the agent's free text. This separation is
what protects us from reward hacking — the model cannot lie about what
happened because the simulator emits the events.
"""

from .events import Event, EventLog, EventType
from .generator import WorldGenerator
from .world import (
    Asset,
    Credential,
    Host,
    Vulnerability,
    World,
)

__all__ = [
    "Asset",
    "Credential",
    "Event",
    "EventLog",
    "EventType",
    "Host",
    "Vulnerability",
    "World",
    "WorldGenerator",
]
