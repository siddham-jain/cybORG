"""Red-side action handlers.

Each handler takes ``(world, args, rng, log)`` and returns a structured
``ActionResult`` dict. Handlers are pure with respect to their arguments —
all randomness flows through ``rng`` so episodes are reproducible.

Handlers do four things:
    1. Validate args against world state (reject unknowns).
    2. Apply the action's effect to ``world`` (mutating its state).
    3. Emit one or more ``Event``s into ``log``.
    4. Return a JSON-friendly result that the agent can read next turn.

These functions deliberately DO NOT compute reward — that lives in the
rubrics, which read from the same event log to keep scoring auditable.
"""

from __future__ import annotations

import random
from typing import Any, Dict

from .events import EventLog, EventType
from .world import World


ActionResult = Dict[str, Any]


def _ok(stdout: str, **extra: Any) -> ActionResult:
    return {"ok": True, "stdout": stdout, "stderr": "", **extra}


def _err(stderr: str) -> ActionResult:
    return {"ok": False, "stdout": "", "stderr": stderr}


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------


def red_scan(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Reveal a host's services to red.

    A scan never fails outright but always raises a low-probability alert so
    blue can sometimes catch reconnaissance. Discovering the same host twice
    is a no-op.
    """

    target = args.get("target")
    host = world.host(target) if target else None
    if host is None:
        return _err(f"unknown host: {target!r}")
    if not _red_can_reach(world, host.id):
        return _err(f"{target} not reachable from current foothold")

    host.discovered = True
    world.red_known_hosts.add(host.id)
    log.emit(EventType.HOST_DISCOVERED, world.step, "red", host=host.id)

    if rng.random() < 0.10:
        log.emit(
            EventType.ALERT_RAISED,
            world.step,
            "red",
            host=host.id,
            kind="scan",
            attribution="red",
        )

    return _ok(
        f"discovered {host.id} ({host.zone}) services={host.services}",
        host=host.id,
        zone=host.zone,
        services=host.services,
        vulnerabilities=[v.id for v in host.vulnerabilities if not v.patched],
    )


# ---------------------------------------------------------------------------
# exploit
# ---------------------------------------------------------------------------


def red_exploit(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Attempt a vulnerability against a discovered host.

    On success the post-condition fires (initial access, priv-esc, credential
    leak, node leak). Detection happens probabilistically per the vuln's
    ``detection_rate``.
    """

    target = args.get("target")
    vuln_id = args.get("vuln")
    host = world.host(target) if target else None
    if host is None or vuln_id is None:
        return _err("exploit requires target and vuln")
    if not host.discovered:
        return _err(f"{target} not yet discovered (run scan first)")
    if not _red_can_reach(world, host.id):
        return _err(f"{target} not reachable")

    vuln = host.vuln(vuln_id)
    if vuln is None or vuln.patched:
        return _err(f"unknown or patched vuln: {vuln_id!r} on {target}")

    log.emit(
        EventType.EXPLOIT_ATTEMPT, world.step, "red", host=host.id, vuln=vuln.id
    )

    if rng.random() > vuln.success_rate:
        log.emit(
            EventType.EXPLOIT_FAILURE, world.step, "red", host=host.id, vuln=vuln.id
        )
        if rng.random() < vuln.detection_rate:
            log.emit(
                EventType.ALERT_RAISED,
                world.step,
                "red",
                host=host.id,
                kind="exploit_failed",
                attribution="red",
            )
        return _err(f"exploit {vuln.id} failed against {host.id}")

    # Success: apply the outcome.
    log.emit(EventType.EXPLOIT_SUCCESS, world.step, "red", host=host.id, vuln=vuln.id)
    if rng.random() < vuln.detection_rate:
        log.emit(
            EventType.ALERT_RAISED,
            world.step,
            "red",
            host=host.id,
            kind="exploit",
            attribution="red",
        )

    return _apply_exploit_outcome(world, host, vuln, log)


def _apply_exploit_outcome(world, host, vuln, log) -> ActionResult:
    out = vuln.outcome
    if out == "access":
        already_owned = host.owned
        host.owned = True
        world.pwned_hosts.add(host.id)
        if not already_owned:
            log.emit(EventType.INITIAL_ACCESS, world.step, "red", host=host.id)
            world.crossed_zones.add(host.zone)
            _tick_red_objective(world, f"initial_access@{host.id}")
        return _ok(f"obtained user-level access on {host.id}", host=host.id)

    if out == "priv_esc":
        if not host.owned:
            return _err("priv_esc requires existing access on host")
        host.privileged = True
        log.emit(EventType.PRIV_ESC, world.step, "red", host=host.id)
        return _ok(f"escalated to privileged on {host.id}", host=host.id)

    if out == "leak_credential" and vuln.leaks_credential:
        cred = world.credential(vuln.leaks_credential)
        if cred is None:
            return _err("leaked credential has been removed from the world")
        world.red_known_creds.add(cred.id)
        log.emit(
            EventType.CRED_OBTAINED,
            world.step,
            "red",
            host=host.id,
            cred=cred.id,
        )
        _tick_red_objective(world, f"obtain_credential@{cred.id}")
        return _ok(
            f"leaked credential {cred.id}",
            host=host.id,
            credential={"id": cred.id, "service": cred.valid_for_service, "host": cred.valid_for_host},
        )

    if out == "leak_node" and vuln.leaks_node:
        next_host = world.host(vuln.leaks_node)
        if next_host is None:
            return _err("leaked node has been removed from the world")
        next_host.discovered = True
        world.red_known_hosts.add(next_host.id)
        log.emit(
            EventType.HOST_DISCOVERED,
            world.step,
            "red",
            host=next_host.id,
            via=host.id,
        )
        return _ok(
            f"discovered new host {next_host.id} via kerberos enumeration",
            host=next_host.id,
        )

    return _ok("exploit succeeded with no specific outcome")


# ---------------------------------------------------------------------------
# use_credential / lateral_move
# ---------------------------------------------------------------------------


def red_use_credential(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Authenticate to a service with a captured credential."""

    cred_id = args.get("cred")
    target = args.get("target")
    service = args.get("service")
    cred = world.credential(cred_id) if cred_id else None
    if cred is None:
        return _err(f"unknown credential {cred_id!r}")
    if cred.id not in world.red_known_creds:
        return _err(f"credential {cred.id} not in red's possession")
    if cred.rotated:
        return _err(f"credential {cred.id} has been rotated by blue")
    host = world.host(target) if target else None
    if host is None:
        return _err(f"unknown host {target!r}")
    if cred.valid_for_host != host.id or cred.valid_for_service != service:
        return _err(
            f"credential {cred.id} is not valid for {host.id}:{service}"
        )
    if not _red_can_reach(world, host.id):
        return _err(f"{host.id} not reachable")

    already_owned = host.owned
    host.owned = True
    world.pwned_hosts.add(host.id)
    log.emit(
        EventType.LATERAL_MOVE,
        world.step,
        "red",
        host=host.id,
        via_cred=cred.id,
    )
    if not already_owned and host.zone not in world.crossed_zones:
        log.emit(EventType.ZONE_CROSSED, world.step, "red", zone=host.zone)
        world.crossed_zones.add(host.zone)
        _tick_red_objective(world, "cross_zone")
    return _ok(f"authenticated to {host.id} via {cred.id}", host=host.id)


def red_lateral_move(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Pivot from one owned host to a reachable next-hop host.

    This is the same as scanning the next host but only succeeds when the
    source host is owned and the target is in the source's outbound firewall
    rules.
    """

    src = args.get("from")
    dst = args.get("to")
    src_host = world.host(src) if src else None
    dst_host = world.host(dst) if dst else None
    if src_host is None or dst_host is None:
        return _err("lateral_move requires from and to host ids")
    if not src_host.owned:
        return _err(f"{src} not owned by red")
    if dst_host.zone not in src_host.firewall_out:
        return _err(f"firewall blocks pivot from {src} to {dst}")
    if dst_host.isolated:
        return _err(f"{dst} has been isolated by blue")
    dst_host.discovered = True
    world.red_known_hosts.add(dst_host.id)
    log.emit(
        EventType.LATERAL_MOVE, world.step, "red", host=dst.id if False else dst_host.id, via=src_host.id
    )
    if dst_host.zone not in world.crossed_zones:
        log.emit(EventType.ZONE_CROSSED, world.step, "red", zone=dst_host.zone)
        world.crossed_zones.add(dst_host.zone)
        _tick_red_objective(world, "cross_zone")
    return _ok(f"pivoted to {dst_host.id} from {src_host.id}", host=dst_host.id)


# ---------------------------------------------------------------------------
# exfiltrate / phish / wait
# ---------------------------------------------------------------------------


def red_exfiltrate(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Pull a data asset from an owned host."""

    target = args.get("target")
    asset_id = args.get("asset")
    host = world.host(target) if target else None
    if host is None or not host.has_asset(asset_id):
        return _err(f"asset {asset_id!r} not found on {target}")
    if not host.owned:
        return _err(f"{host.id} not owned by red")
    if asset_id in world.exfiltrated_assets:
        return _err(f"{asset_id} already exfiltrated")
    world.exfiltrated_assets.add(asset_id)
    log.emit(
        EventType.EXFIL_SUCCESS, world.step, "red", host=host.id, asset=asset_id
    )
    if rng.random() < 0.4:  # exfil is noisy by nature
        log.emit(
            EventType.ALERT_RAISED,
            world.step,
            "red",
            host=host.id,
            kind="exfil",
            attribution="red",
        )
    _tick_red_objective(world, f"exfiltrate@{asset_id}")
    return _ok(f"exfiltrated {asset_id} from {host.id}", asset=asset_id)


def red_phish(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Send a phishing lure to all NPCs of a given role.

    Success on any one NPC discovers their home host. This gives red a
    cheap-but-noisy way to bootstrap when scanning is failing.
    """

    role = args.get("role")
    if role is None:
        return _err("phish requires a role")
    targets = [n for n in world.npcs.values() if n.role == role]
    if not targets:
        return _err(f"no NPCs with role {role}")
    for npc in targets:
        if rng.random() < npc.susceptibility:
            host = world.host(npc.home_host)
            if host is not None:
                host.discovered = True
                world.red_known_hosts.add(host.id)
                log.emit(
                    EventType.PHISH_SUCCESS,
                    world.step,
                    "red",
                    npc=npc.id,
                    host=host.id,
                )
                # Phishing is loud.
                if rng.random() < 0.5:
                    log.emit(
                        EventType.ALERT_RAISED,
                        world.step,
                        "red",
                        host=host.id,
                        kind="phish",
                        attribution="red",
                    )
                return _ok(
                    f"phish landed on {npc.id}; discovered home host {host.id}",
                    host=host.id,
                )
    log.emit(EventType.PHISH_FAILURE, world.step, "red", role=role)
    return _err(f"phishing {role} failed (no susceptible NPC bit)")


def red_wait(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Burn a turn deliberately.

    Useful as a cooldown when blue is actively monitoring. Counts toward the
    step budget and toward the diversity audit (so spamming wait is bad).
    """

    return _ok("idle")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _red_can_reach(world: World, host_id: str) -> bool:
    """True when red has any owned host that can reach ``host_id``.

    The DMZ is always reachable from "outside" so an empty foothold can
    still scan/exploit DMZ hosts.
    """

    target = world.host(host_id)
    if target is None or target.isolated:
        return False
    if "dmz" in target.firewall_in:
        return True
    for owned_id in world.pwned_hosts:
        owned = world.host(owned_id)
        if owned is None:
            continue
        if target.zone in owned.firewall_out:
            return True
    return False


def _tick_red_objective(world: World, predicate: str) -> None:
    if predicate in world.red_objectives and predicate not in world.red_objectives_done:
        world.red_objectives_done.add(predicate)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

RED_DISPATCH = {
    "scan": red_scan,
    "exploit": red_exploit,
    "use_credential": red_use_credential,
    "lateral_move": red_lateral_move,
    "exfiltrate": red_exfiltrate,
    "phish": red_phish,
    "wait": red_wait,
}
