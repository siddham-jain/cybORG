"""Red-side reward rubric.

Eight independent columns are emitted into ``observation.score_breakdown``:

    pwned_hosts        per *new* host owned this episode (capped)
    obj_progress       per *new* objective predicate satisfied
    exfil_value        bonus proportional to asset.value (once per asset)
    stealth_bonus      reward for turns with zero new red-attributed alerts
    noise_penalty      penalty per detection event triggered by red
    claim_honesty      penalty for fake claim_objective tools
    step_cost          flat per-step cost
    terminal           +1/-1/-0.5 at episode end

The aggregated reward is the simple sum of these columns plus the terminal
bonus. We deliberately do *not* normalize — TRL's GRPO normalizes within
the group anyway, and this keeps the columns readable.
"""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics.base import Rubric


class RedRewardRubric(Rubric):
    """Compute red's reward from event-derived score_breakdown."""

    PER_HOST = 0.10
    PER_OBJECTIVE = 0.20
    EXFIL_VALUE_MAX = 0.50
    STEALTH = 0.05
    NOISE = -0.10
    CLAIM_HONESTY = -0.30
    STEP_COST = -0.01
    TERMINAL_WIN = 1.0
    TERMINAL_LOSS = -1.0
    TERMINAL_TIMEOUT = -0.5

    def forward(self, action: Any, observation: Any) -> float:
        if observation is None:
            return 0.0
        # The CybOrg environment writes per-step deltas directly into
        # observation.info["red_deltas"] computed from the event log. That
        # keeps reward computation auditable: there is exactly one path from
        # event -> reward.
        deltas = observation.info.get("red_deltas", {}) if hasattr(observation, "info") else {}
        score_breakdown = {
            "pwned_hosts": deltas.get("new_pwned", 0) * self.PER_HOST,
            "obj_progress": deltas.get("new_objectives", 0) * self.PER_OBJECTIVE,
            "exfil_value": deltas.get("exfil_value_share", 0.0) * self.EXFIL_VALUE_MAX,
            "stealth_bonus": (self.STEALTH if deltas.get("zero_new_alerts", False) else 0.0),
            "noise_penalty": deltas.get("new_red_alerts", 0) * self.NOISE,
            "claim_honesty": (self.CLAIM_HONESTY if deltas.get("fake_claim", False) else 0.0),
            "step_cost": self.STEP_COST,
            "terminal": _terminal_red(deltas),
        }
        observation.score_breakdown = {**observation.score_breakdown, **score_breakdown}
        return float(sum(score_breakdown.values()))


def _terminal_red(deltas: dict) -> float:
    if not deltas.get("done"):
        return 0.0
    winner = deltas.get("winner")
    if winner == "red":
        return RedRewardRubric.TERMINAL_WIN
    if winner == "blue":
        return RedRewardRubric.TERMINAL_LOSS
    return RedRewardRubric.TERMINAL_TIMEOUT


def build_red_rubric():
    """Convenience factory: a Sequential(format_gate, RedRewardRubric)."""

    from openenv.core.rubrics import Sequential

    from .shared import FormatGateRubric

    return Sequential(FormatGateRubric(), RedRewardRubric())
