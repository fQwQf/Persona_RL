#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Apply transparent rules to common-schema prediction JSONL."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from persona_rl.results import PredictionRecord, ScoreRecord, read_models, write_models
from persona_rl.schema import parse_jsonl
from persona_rl.scoring import score_prediction

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    predictions: Path,
    scenarios: Path,
    output: Path = Path("artifacts/scores.jsonl"),
    smoke_only: bool = False,
) -> None:
    """Run the explicitly labelled smoke scorer; use llm_score_outputs for research scores."""
    if not smoke_only:
        raise typer.BadParameter(
            "keyword scoring is smoke-only; pass --smoke-only or use llm_score_outputs.py"
        )
    scenario_map = {scenario.id: scenario for scenario in parse_jsonl(str(scenarios))}
    records = read_models(str(predictions), PredictionRecord)
    scores: list[ScoreRecord] = []
    for prediction in records:
        scenario = scenario_map.get(prediction.scenario_id)
        if scenario is None:
            raise typer.BadParameter(
                f"prediction references unknown scenario {prediction.scenario_id}"
            )
        scores.append(score_prediction(prediction, scenario))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_models(str(output), scores)
    output.with_name(f"{output.stem}.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "rule-score-smoke-v1",
                "prediction_path": str(predictions),
                "prediction_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                "scenario_path": str(scenarios),
                "scenario_sha256": hashlib.sha256(scenarios.read_bytes()).hexdigest(),
                "rubric_version": "rubric-v1",
                "n_scores": len(scores),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    typer.echo(f"scored {len(scores)} predictions -> {output}")


if __name__ == "__main__":
    app()
