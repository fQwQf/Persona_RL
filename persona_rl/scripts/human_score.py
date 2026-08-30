#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12", "rich>=13.7"]
# ///

"""Blind, low-volume human scoring for LLM-judge external-validity audits."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import typer

from persona_rl.results import HumanScoreRecord, PredictionRecord, read_models, write_models
from persona_rl.schema import parse_jsonl

app = typer.Typer(no_args_is_help=True)


def _score(label: str) -> int:
    """Ask for a bounded Likert score."""
    value = typer.prompt(label, type=int)
    if not 1 <= value <= 5:
        raise typer.BadParameter("scores must be integers from 1 to 5")
    return value


@app.command()
def main(
    predictions: Path,
    scenarios: Path = Path("data/raw/scenarios.jsonl"),
    output: Path = Path("artifacts/report/human_scores.jsonl"),
    limit: int = 100,
    seed: int = 7,
) -> None:
    """Score shuffled responses without revealing the model method."""
    rows = read_models(str(predictions), PredictionRecord)
    if not rows:
        raise typer.BadParameter("prediction JSONL is empty")
    rng = random.Random(seed)
    selected = rows[:limit]
    rng.shuffle(selected)
    reviewer = os.environ.get("PERSONA_RL_REVIEWER", "anonymous")
    scenario_map = {row.id: row for row in parse_jsonl(str(scenarios))} if scenarios.exists() else {}
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_models(str(output), HumanScoreRecord) if output.exists() else []
    seen = {row.item_id for row in existing}
    scored = list(existing)
    for row in selected:
        item_id = hashlib.sha256(
            f"{row.scenario_id}:{row.sample_index}:{row.prompt_variant}".encode()
        ).hexdigest()[:16]
        if item_id in seen:
            continue
        scenario = scenario_map.get(row.scenario_id)
        context = ""
        if scenario is not None:
            context = (
                f"Situation: {scenario.situation}\n"
                f"Behavior rubric: {'; '.join(scenario.behavior_rubric)}\n"
                f"Options: {' | '.join(scenario.hidden_task.options)}\n"
            )
        typer.echo(f"\n[{item_id}] Scenario {row.scenario_id}\n{context}Response:\n{row.response}\n")
        result = HumanScoreRecord(
            item_id=item_id,
            method_code=f"system-{item_id}",
            scenario_id=row.scenario_id,
            trait_fidelity=_score("Trait fidelity (1-5)"),
            behavior_validity=_score("Behavior validity (1-5)"),
            truthfulness=_score("Truthfulness (1-5)"),
            safety=_score("Safety (1-5)"),
            confidence=_score("Confidence (1-5)"),
            note=typer.prompt("Note", default=""),
            reviewer=reviewer,
        )
        scored.append(result)
    write_models(str(output), scored)
    typer.echo(f"scored {len(scored) - len(existing)} new items -> {output}")


if __name__ == "__main__":
    app()
