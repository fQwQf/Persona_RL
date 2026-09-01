"""Typed records for predictions, scores, runs, and human audit."""

from __future__ import annotations

import json
from datetime import datetime
try:
    from datetime import UTC
except ImportError:  # Python 3.10 compatibility for shared research servers.
    from datetime import timezone
    UTC = timezone.utc
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

MethodName = Literal[
    "base",
    "prompt_only",
    "sft",
    "direct_dpo",
    "kto",
    "personality_vector",
    "machine_mindset",
    "persllm",
    "rolellm",
    "personagym",
    "representation_engineering",
    "big5_chat",
    "simpo",
    "pc_dpo",
]


def parse_method(value: str) -> MethodName:
    """Parse a CLI method name into the closed method vocabulary."""
    match value:
        case (
            "base"
            | "prompt_only"
            | "sft"
            | "direct_dpo"
            | "kto"
            | "personality_vector"
            | "machine_mindset"
            | "persllm"
            | "rolellm"
            | "personagym"
            | "representation_engineering"
            | "big5_chat"
            | "simpo"
            | "pc_dpo"
        ):
            return value
        case _:
            raise ValueError(f"unknown method: {value}")


class RunManifest(BaseModel):
    """Immutable provenance for one baseline or proposed-method run."""

    model_config = ConfigDict(frozen=True)
    run_id: str = Field(min_length=1)
    method: MethodName
    model_id: str = Field(min_length=1)
    source_repo: str = ""
    source_commit: str = ""
    command: str = Field(min_length=1)
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    seed: int
    config: dict[str, str] = Field(default_factory=dict)


class PredictionRecord(BaseModel):
    """One model response in the common cross-baseline schema."""

    model_config = ConfigDict(frozen=True)
    run_id: str = Field(min_length=1)
    method: MethodName
    model_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    family: str = Field(min_length=1)
    counterfactual_group: str = ""
    target: dict[str, int]
    target_intensity: float = Field(default=1.0, ge=0, le=1)
    prompt_variant: str = "canonical"
    style_family: str = "neutral"
    language: str = "zh"
    temperature: float = Field(ge=0, le=2)
    sample_index: int = Field(ge=0)
    prompt: str = Field(min_length=1)
    response: str = ""
    latency_ms: float = Field(ge=0)


class ScoreRecord(BaseModel):
    """Automatic scores joined to a prediction."""

    model_config = ConfigDict(frozen=True)
    prediction: PredictionRecord
    trait_fidelity: float = Field(ge=0, le=1)
    behavior_validity: float = Field(ge=0, le=1)
    truthfulness: float = Field(ge=0, le=1)
    safety: float = Field(ge=0, le=1)
    sycophancy: float = Field(ge=0, le=1)
    capability_retention: float | None = Field(default=None, ge=0, le=1)
    rule_flags: tuple[str, ...] = ()
    judge_model: str = "rules-v1"
    judge_confidence: float = Field(ge=0, le=1)
    rubric_version: str = "rubric-v1"
    trait_scores: dict[str, float] = Field(default_factory=dict)
    selected_option: str = ""
    gold_option_match: float | None = Field(default=None, ge=0, le=1)
    behavior_probability: float | None = Field(default=None, ge=0, le=1)


class AuditRecord(BaseModel):
    """Human audit decision for a sampled prediction or scenario."""

    model_config = ConfigDict(frozen=True)
    item_id: str = Field(min_length=1)
    decision: Literal["accept", "reject", "uncertain"]
    score: int = Field(ge=1, le=5)
    note: str = ""
    reviewer: str = "anonymous"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class HumanScoreRecord(BaseModel):
    """Blind human rubric score for external validity auditing."""

    model_config = ConfigDict(frozen=True)
    item_id: str = Field(min_length=1)
    method_code: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    trait_fidelity: int = Field(ge=1, le=5)
    behavior_validity: int = Field(ge=1, le=5)
    truthfulness: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    confidence: int = Field(ge=1, le=5)
    note: str = ""
    reviewer: str = "anonymous"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


ModelT = TypeVar("ModelT", bound=BaseModel)


def write_models(path: str, records: list[BaseModel]) -> None:
    """Write validated Pydantic models as UTF-8 JSONL."""
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            )
            handle.write("\n")


def read_models(path: str, model_type: type[ModelT]) -> list[ModelT]:
    """Read and validate JSONL records at a trust boundary."""
    records: list[ModelT] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model_type.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid record at line {line_number}: {exc}") from exc
    return records
