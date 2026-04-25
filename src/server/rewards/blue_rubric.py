"""Blue-side reward rubric.

Mirrors the red rubric in shape. Blue wins by preventing red from
exfiltrating the crown jewel within the step budget; loses if red gets it.
"""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics.base import Rubric


class BlueRewardRubric(Rubric):
    """Compute blue's reward from event-derived score_breakdown."""

    DETECTION = 0.15
    CONTAINMENT = 0.30
    CONTINUITY = -0.20
    FALSE_FINDING = -0.20
    OVER_BLOCK = -0.15
    STEP_COST = -0.01
    TERMINAL_WIN = 1.0
    TERMINAL_LOSS = -1.0
    TERMINAL_TIMEOUT = 0.5  # blue gets a small reward for surviving

    def forward(self, action: Any, observation: Any) -> float:
        if observation is None:
            return 0.0
        deltas = observation.info.get("blue_deltas", {}) if hasattr(observation, "info") else {}
        breakdown = {
            "detection_credit": deltas.get("new_findings_true", 0) * self.DETECTION,
            "containment_credit": deltas.get("new_containments", 0) * self.CONTAINMENT,
            "continuity": deltas.get("continuity_drop", 0.0) * self.CONTINUITY,
            "false_finding": deltas.get("new_findings_false", 0) * self.FALSE_FINDING,
            "over_block": deltas.get("new_overblocks", 0) * self.OVER_BLOCK,
            "step_cost": self.STEP_COST,
            "terminal": _terminal_blue(deltas),
        }
        observation.score_breakdown = {**observation.score_breakdown, **breakdown}
        return float(sum(breakdown.values()))


def _terminal_blue(deltas: dict) -> float:
    if not deltas.get("done"):
        return 0.0
    winner = deltas.get("winner")
    if winner == "blue":
        return BlueRewardRubric.TERMINAL_WIN
    if winner == "red":
        return BlueRewardRubric.TERMINAL_LOSS
    return BlueRewardRubric.TERMINAL_TIMEOUT


def build_blue_rubric():
    """Convenience factory: a Sequential(format_gate, BlueRewardRubric)."""

    from openenv.core.rubrics import Sequential

    from .shared import FormatGateRubric

    return Sequential(FormatGateRubric(), BlueRewardRubric())
