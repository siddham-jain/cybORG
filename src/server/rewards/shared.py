"""Shared rubric primitives.

Includes:
    * ``FormatGateRubric`` — returns 1.0 if the action validated, 0.0 otherwise.
      When wrapped in ``Sequential`` it short-circuits the entire rubric and
      lets us safely zero-out malformed actions without crashing the run.
"""

from __future__ import annotations

from typing import Any

from openenv.core.rubrics.base import Rubric


class FormatGateRubric(Rubric):
    """Hard gate on action JSON validity.

    The CybOrg environment marks malformed actions by setting
    ``observation.info["format_ok"] = False``. This rubric just lifts that
    flag into a 0/1 reward so it can sit at the head of a ``Sequential``
    chain and short-circuit reward emission for the whole step.
    """

    def forward(self, action: Any, observation: Any) -> float:
        if observation is None or not hasattr(observation, "info"):
            return 1.0
        return 1.0 if observation.info.get("format_ok", True) else 0.0
