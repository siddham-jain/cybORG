"""Reward rubrics for CybOrg.

Rubrics are stateful so they can compare turn N vs turn N-1 (i.e. credit a
*new* host owned, not the same host every turn). All rubrics read from the
event log via the observation's ``info["events"]`` payload — they never
parse free text.

The top-level :func:`build_red_rubric` and :func:`build_blue_rubric`
factories return ready-to-use ``Sequential`` rubrics that gate on JSON
validity first, then run the WeightedSum of the eight independent reward
columns we monitor in training dashboards.
"""

from .red_rubric import RedRewardRubric, build_red_rubric
from .blue_rubric import BlueRewardRubric, build_blue_rubric
from .shared import FormatGateRubric

__all__ = [
    "RedRewardRubric",
    "BlueRewardRubric",
    "FormatGateRubric",
    "build_red_rubric",
    "build_blue_rubric",
]
