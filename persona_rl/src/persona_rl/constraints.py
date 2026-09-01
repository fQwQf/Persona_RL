"""Constraint-aware preference weights and differentiable PC-DPO loss."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from typing import TypedDict


class JudgeRewardPayload(TypedDict, total=False):
    """JSON reward fields emitted by the independent judge."""

    trait_score: float
    criterion_score: float
    invariance_score: float
    truth_score: float
    safety_score: float
    uncertainty: float
    rejected: bool


@dataclass(frozen=True, slots=True)
class ConstraintReward:
    """Cached independent-judge heads used to construct PC-DPO pairs."""

    trait: float
    criterion: float
    invariance: float
    truth: float
    safety: float
    uncertainty: float
    rejected: bool = False

    def weight(self) -> float:
        """Return a bounded weight that downweights uncertain or unsafe pairs."""
        if self.rejected or self.truth < 0.7 or self.safety < 0.7:
            return 0.0
        quality = (self.trait + self.criterion + self.invariance) / 3
        return max(0.05, min(2.0, quality * (1.0 - self.uncertainty)))


def pc_dpo_scalar_loss(
    chosen_log_ratio: float,
    rejected_log_ratio: float,
    reward: ConstraintReward,
    beta: float = 0.1,
) -> float:
    """Compute the scalar PC-DPO loss for deterministic unit tests and logging."""
    margin = beta * (chosen_log_ratio - rejected_log_ratio)
    dpo = -log(1.0 / (1.0 + exp(-margin)))
    constraint = (1.0 - reward.criterion) + (1.0 - reward.invariance)
    uncertainty = reward.uncertainty
    return reward.weight() * dpo + 0.25 * constraint + 0.25 * uncertainty


def reward_from_judge(payload: JudgeRewardPayload) -> ConstraintReward:
    """Parse judge heads at the data boundary into a typed reward object."""
    def value(name: str, alias: str, default: float) -> float:
        raw = payload.get(name, payload.get(alias, default))
        return float(raw)
    return ConstraintReward(
        trait=value("trait_score", "trait", 0.0),
        criterion=value("criterion_score", "criterion", 0.0),
        invariance=value("invariance_score", "invariance", 0.0),
        truth=value("truth_score", "truth", 0.0),
        safety=value("safety_score", "safety", 0.0),
        uncertainty=float(payload.get("uncertainty", 1.0)),
        rejected=bool(payload.get("rejected", False)),
    )
