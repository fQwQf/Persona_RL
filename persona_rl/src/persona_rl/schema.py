"""Typed dataset records and JSONL boundary parsing."""

from __future__ import annotations

import json
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

TraitName = Literal["conscientiousness", "agreeableness", "honesty_humility"]
SplitName = Literal["train", "validation", "test", "audit"]


class TargetTraits(BaseModel):
    """Three-dimensional target trait vector."""

    model_config = ConfigDict(frozen=True)
    conscientiousness: int = Field(ge=-1, le=1)
    agreeableness: int = Field(ge=-1, le=1)
    honesty_humility: int = Field(ge=-1, le=1)


class HiddenTask(BaseModel):
    """Machine-checkable downstream behavioral task."""

    model_config = ConfigDict(frozen=True)
    type: Literal["binary_decision", "structured_check"]
    question: str = ""
    gold_behavior: int = Field(ge=0, le=1)
    options: tuple[str, ...] = ()
    gold_option: str = ""
    target_trait: TraitName | None = None


class Scenario(BaseModel):
    """A single scenario crossing the JSONL trust boundary."""

    model_config = ConfigDict(frozen=True)
    id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    split: SplitName
    target_z: TargetTraits
    target_intensity: float = Field(default=1.0, ge=0, le=1)
    situation: str = Field(min_length=1)
    history: tuple[str, ...] = ()
    hard_constraints: tuple[str, ...] = ()
    behavior_rubric: tuple[str, ...] = ()
    forbidden_behaviors: tuple[str, ...] = ()
    capability_rubric: tuple[str, ...] = ()
    evaluation_tags: tuple[str, ...] = ()
    hidden_task: HiddenTask
    counterfactual_group: str = Field(min_length=1)
    style_family: str = "neutral"


SCHEMA_VERSION: Final[str] = "scenario-v1"


def parse_jsonl(path: str) -> list[Scenario]:
    """Parse and validate scenario JSONL records."""
    records: list[Scenario] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(Scenario.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid scenario at line {line_number}: {exc}") from exc
    return records


def write_jsonl(path: str, records: list[Scenario]) -> None:
    """Write validated scenarios as stable UTF-8 JSONL."""
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            )
            handle.write("\n")
