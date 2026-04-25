"""Blue-side action handlers.

Symmetric to ``red_actions`` — handlers mutate world state and emit events
that the rubrics consume. Blue's actions are intentionally costly (over-block
penalty, false-finding penalty, continuity penalty) so the model has to
*think* before isolating everything.
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
# monitor / analyze_alert
# ---------------------------------------------------------------------------


def blue_monitor(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Place a sensor on a host so future events fire ALERT_RAISED reliably.

    Once a host is monitored, any red exploit on that host raises an alert
    with probability 1.0 instead of the vuln's intrinsic ``detection_rate``.
    """

    host = world.host(args.get("host", ""))
    if host is None:
        return _err("monitor requires a valid host id")
    host.monitored = True
    return _ok(f"monitor enabled on {host.id}", host=host.id)


def blue_analyze_alert(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Inspect an alert. Returns the underlying event so blue can decide.

    Analyzing a benign (green-NPC) alert costs nothing; analyzing a true
    alert is the expected precondition for ``submit_finding``.
    """

    alert_id = args.get("alert_id")
    try:
        idx = int(alert_id)
        ev = log.events[idx]
    except (ValueError, TypeError, IndexError):
        return _err(f"unknown alert_id {alert_id!r}")
    if ev.type not in {EventType.ALERT_RAISED, EventType.ALERT_BENIGN}:
        return _err(f"event {alert_id} is not an alert")
    return _ok(
        f"alert {alert_id} on {ev.data.get('host')} is "
        f"{'malicious' if ev.type == EventType.ALERT_RAISED else 'benign'}",
        attribution=ev.data.get("attribution", "green" if ev.type == EventType.ALERT_BENIGN else "red"),
        kind=ev.data.get("kind"),
        host=ev.data.get("host"),
    )


# ---------------------------------------------------------------------------
# block_ip / isolate_host / patch / rotate_cred
# ---------------------------------------------------------------------------


def blue_block_ip(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Drop a host from the inbound firewall on every other host.

    Cheap but visible; if the host had no malicious activity in the recent
    window the rubric fires an OVERBLOCK penalty.
    """

    host = world.host(args.get("host", ""))
    if host is None:
        return _err("block_ip requires a valid host id")
    for other in world.hosts.values():
        other.firewall_in.discard(host.zone)
    log.emit(EventType.IP_BLOCKED, world.step, "blue", host=host.id)
    if not _host_was_malicious_recently(world, log, host.id):
        log.emit(EventType.OVERBLOCK, world.step, "blue", host=host.id)
        world.blue_overblocks += 1
    return _ok(f"blocked traffic from {host.id}", host=host.id)


def blue_isolate_host(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Cut a host off the network entirely.

    Stops red from interacting with that host but takes the host out of the
    SLA — counted in the continuity penalty.
    """

    host = world.host(args.get("host", ""))
    if host is None:
        return _err("isolate_host requires a valid host id")
    if host.isolated:
        return _err(f"{host.id} already isolated")
    host.isolated = True
    log.emit(EventType.HOST_ISOLATED, world.step, "blue", host=host.id)
    if not _host_was_malicious_recently(world, log, host.id):
        log.emit(EventType.OVERBLOCK, world.step, "blue", host=host.id)
        world.blue_overblocks += 1
    return _ok(f"isolated {host.id}", host=host.id)


def blue_patch(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Patch a specific vulnerability on a host."""

    host = world.host(args.get("host", ""))
    vuln_id = args.get("vuln")
    if host is None:
        return _err("patch requires a valid host id")
    vuln = host.vuln(vuln_id) if vuln_id else None
    if vuln is None:
        return _err(f"unknown vuln {vuln_id!r} on {host.id}")
    vuln.patched = True
    log.emit(EventType.HOST_PATCHED, world.step, "blue", host=host.id, vuln=vuln.id)
    return _ok(f"patched {vuln.id} on {host.id}")


def blue_rotate_cred(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Rotate a credential, invalidating any captured copies."""

    cred = world.credential(args.get("cred", ""))
    if cred is None:
        return _err("rotate_cred requires a valid cred id")
    cred.rotated = True
    log.emit(EventType.CRED_ROTATED, world.step, "blue", cred=cred.id)
    return _ok(f"rotated credential {cred.id}")


# ---------------------------------------------------------------------------
# submit_finding
# ---------------------------------------------------------------------------


def blue_submit_finding(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    """Officially attribute a host+vuln to the red actor.

    True positives credit ``detection_credit``; false positives incur the
    ``false_finding`` penalty. Each (host, vuln) pair can only be claimed
    once per episode.
    """

    host = world.host(args.get("host", ""))
    vuln_id = args.get("vuln")
    actor = args.get("actor", "red")
    if host is None or vuln_id is None:
        return _err("submit_finding requires host, vuln, actor")
    finding_key = f"{host.id}:{vuln_id}"
    if finding_key in world.blue_findings or finding_key in {
        f for f in _seen_false(world)
    }:
        return _err(f"finding {finding_key} already submitted")

    # Truth check: did the simulator actually emit a successful exploit on
    # this host:vuln pair attributable to red?
    truth = any(
        e.type == EventType.EXPLOIT_SUCCESS
        and e.data.get("host") == host.id
        and e.data.get("vuln") == vuln_id
        for e in log.events
    )
    if truth and actor == "red":
        world.blue_findings.add(finding_key)
        log.emit(
            EventType.FINDING_TRUE,
            world.step,
            "blue",
            host=host.id,
            vuln=vuln_id,
        )
        return _ok(f"finding accepted on {host.id}:{vuln_id}")
    log.emit(
        EventType.FINDING_FALSE,
        world.step,
        "blue",
        host=host.id,
        vuln=vuln_id,
    )
    world.blue_false_findings += 1
    return _err(f"finding rejected for {host.id}:{vuln_id} (no matching event)")


def blue_wait(
    world: World, args: Dict[str, Any], rng: random.Random, log: EventLog
) -> ActionResult:
    return _ok("idle")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _host_was_malicious_recently(world, log, host_id, window=5) -> bool:
    """True if any red event involving ``host_id`` happened in the last
    ``window`` turns.
    """

    cutoff = max(0, world.step - window)
    for ev in reversed(log.events):
        if ev.turn < cutoff:
            break
        if ev.role != "red":
            continue
        if ev.data.get("host") == host_id and ev.type in {
            EventType.EXPLOIT_ATTEMPT,
            EventType.EXPLOIT_SUCCESS,
            EventType.LATERAL_MOVE,
            EventType.HOST_DISCOVERED,
            EventType.EXFIL_SUCCESS,
        }:
            return True
    return False


def _seen_false(world) -> set:
    return getattr(world, "_false_keys", set())


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

BLUE_DISPATCH = {
    "monitor": blue_monitor,
    "analyze_alert": blue_analyze_alert,
    "block_ip": blue_block_ip,
    "isolate_host": blue_isolate_host,
    "patch": blue_patch,
    "rotate_cred": blue_rotate_cred,
    "submit_finding": blue_submit_finding,
    "wait": blue_wait,
}
