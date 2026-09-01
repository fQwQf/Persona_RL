#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Filter candidate pairs with an OpenAI-compatible JSON judge."""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Literal

import typer
from pydantic import BaseModel, ConfigDict, Field, field_validator

app = typer.Typer(no_args_is_help=True)


class JudgeResult(BaseModel):
    """Validated pairwise judge output used to filter training pairs."""

    model_config = ConfigDict(frozen=True, extra="ignore")
    winner: Literal["a", "b", "tie"]
    trait_score: float = Field(ge=0, le=1)
    truth_score: float = Field(ge=0, le=1)
    safety_score: float = Field(ge=0, le=1)
    criterion_score: float = Field(default=0.5, ge=0, le=1)
    invariance_score: float = Field(default=0.5, ge=0, le=1)
    uncertainty: float = Field(default=0.0, ge=0, le=1)

    @field_validator("winner", mode="before")
    @classmethod
    def normalize_winner(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower().replace(" ", "_")
            aliases = {
                "a": "a", "b": "b", "tie": "tie", "tied": "tie", "draw": "tie", "equal": "tie", "same": "tie",
                "candidate_a": "a", "candidate_b": "b",
                "answer_a": "a", "answer_b": "b", "option_a": "a", "option_b": "b",
            }
            if normalized in aliases:
                return aliases[normalized]
            if "candidate_a" in normalized or "answer_a" in normalized or "option_a" in normalized:
                return "a"
            if "candidate_b" in normalized or "answer_b" in normalized or "option_b" in normalized:
                return "b"
            if any(token in normalized for token in ("tie", "draw", "equal", "same")):
                return "tie"
            return value
        return value


class PairRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")
    id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    candidate_a: str
    candidate_b: str
    family: str = ""
    target_z: dict[str, int] = Field(default_factory=dict)
    counterfactual_group: str = ""


def _judge(endpoint: str, key: str, model: str, payload: dict[str, object]) -> JudgeResult:
    prompt = (
        "Return JSON with winner (a/b/tie), trait_score (0-1), truth_score (0-1), "
        "safety_score (0-1), criterion_score (0-1), invariance_score (0-1), "
        "uncertainty (0-1). Do not explain.\n" + json.dumps(payload, ensure_ascii=False)
    )
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content = json.load(response)["choices"][0]["message"]["content"]
    return JudgeResult.model_validate_json(content)


@app.command()
def main(
    input_path: Path,
    output_path: Path = Path("data/processed/judged_pairs.jsonl"),
    model: str = "gpt-4o-mini",
    second_model: str = "",
    second_base_url: str = "",
    min_trait_score: float = 0.7,
    min_truth_score: float = 0.7,
    min_safety_score: float = 0.7,
    audit_path: Path | None = None,
) -> None:
    """Judge candidate rows and emit only decisive preferred/rejected pairs."""
    key = os.environ.get("OPENAI_API_KEY", "")
    endpoint = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    if not key:
        raise typer.BadParameter("set OPENAI_API_KEY or use a local compatible endpoint")
    if not all(0 <= value <= 1 for value in (min_trait_score, min_truth_score, min_safety_score)):
        raise typer.BadParameter("score thresholds must be between zero and one")
    rows = [
        PairRow.model_validate_json(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_destination = audit_path or output_path.with_name(f"{output_path.stem}.audit.jsonl")
    audit_destination.parent.mkdir(parents=True, exist_ok=True)
    accepted_count = 0
    with (
        output_path.open("w", encoding="utf-8") as handle,
        audit_destination.open("w", encoding="utf-8") as audit_handle,
    ):
        for row in rows:
            payload = {
                "prompt": row.prompt,
                "candidate_a": row.candidate_a,
                "candidate_b": row.candidate_b,
                "family": row.family,
                "target_z": row.target_z,
                "counterfactual_group": row.counterfactual_group,
            }
            judged = _judge(
                endpoint,
                key,
                model,
                payload,
            )
            secondary_endpoint = second_base_url or os.environ.get("OPENAI_SECOND_BASE_URL", endpoint)
            second = _judge(secondary_endpoint, key, second_model, payload) if second_model else None
            reasons: list[str] = []
            verdicts = [("primary", judged)]
            if second is not None:
                verdicts.append(("secondary", second))
            for label, verdict in verdicts:
                if verdict.winner == "tie":
                    reasons.append(f"{label}_tie")
                if verdict.trait_score < min_trait_score:
                    reasons.append(f"{label}_trait_threshold")
                if verdict.truth_score < min_truth_score:
                    reasons.append(f"{label}_truth_threshold")
                if verdict.safety_score < min_safety_score:
                    reasons.append(f"{label}_safety_threshold")
                if verdict.criterion_score < min_trait_score:
                    reasons.append(f"{label}_criterion_threshold")
                if verdict.invariance_score < min_trait_score:
                    reasons.append(f"{label}_invariance_threshold")
                if verdict.uncertainty > 1 - min_truth_score:
                    reasons.append(f"{label}_uncertainty_threshold")
            if second is not None and second.winner != judged.winner:
                reasons.append("judge_disagreement")
            accepted = not reasons
            audit_handle.write(
                json.dumps(
                    {
                        "id": row.id,
                        "accepted": accepted,
                        "reasons": reasons,
                        "primary_model": model,
                        "rubric_version": "pair-rubric-v1",
                        "primary": judged.model_dump(mode="json"),
                        "secondary_model": second_model or None,
                        "secondary_endpoint": secondary_endpoint if second_model else None,
                        "secondary": second.model_dump(mode="json") if second else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if not accepted:
                continue
            accepted_count += 1
            winner, loser = (
                (row.candidate_a, row.candidate_b)
                if judged.winner == "a"
                else (row.candidate_b, row.candidate_a)
            )
            handle.write(
                json.dumps(
                    {
                        "prompt": row.prompt,
                        "chosen": winner,
                        "rejected": loser,
                        "id": row.id,
                        "rubric_version": "pair-rubric-v1",
                        "judge": judged.model_dump(mode="json"),
                        "pc_reward": {
                            "trait": judged.trait_score,
                            "criterion": judged.criterion_score,
                            "invariance": judged.invariance_score,
                            "truth": judged.truth_score,
                            "safety": judged.safety_score,
                            "uncertainty": judged.uncertainty,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    typer.echo(
        f"judged {len(rows)} rows; accepted {accepted_count} -> {output_path}; "
        f"audit -> {audit_destination}"
    )


if __name__ == "__main__":
    app()
