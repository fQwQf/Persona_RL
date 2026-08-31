#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Normalize official baseline outputs into the common prediction schema."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import typer
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from persona_rl.results import PredictionRecord, parse_method, write_models
from persona_rl.schema import parse_jsonl

app = typer.Typer(no_args_is_help=True)


class ExternalRow(BaseModel):
    """Canonical external adapter row before scenario metadata is joined."""

    model_config = ConfigDict(frozen=True)
    scenario_id: str = Field(validation_alias=AliasChoices("scenario_id", "id"))
    response: str = Field(validation_alias=AliasChoices("response", "text", "output"))
    method: str = ""
    model_id: str = ""
    temperature: float = Field(default=0.7, ge=0, le=2)
    sample_index: int | None = Field(default=None, ge=0)
    prompt_variant: str = "canonical"


def _rows(path: Path, delimiter: str) -> list[ExternalRow]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8", newline="") as handle:
            return [
                ExternalRow.model_validate(row)
                for row in csv.DictReader(handle, delimiter=delimiter)
            ]
    return [
        ExternalRow.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@app.command()
def main(
    raw: Path,
    scenarios: Path,
    output: Path = Path("artifacts/external_predictions.jsonl"),
    delimiter: str = ",",
    source_manifest: Path | None = None,
    method_override: str = "",
    model_override: str = "",
) -> None:
    """Join official raw outputs to scenario metadata and preserve provenance."""
    scenario_map = {scenario.id: scenario for scenario in parse_jsonl(str(scenarios))}
    predictions: list[PredictionRecord] = []
    for index, row in enumerate(_rows(raw, delimiter)):
        scenario = scenario_map.get(row.scenario_id)
        if scenario is None:
            raise typer.BadParameter(f"unknown scenario id: {row.scenario_id}")
        method_value = row.method or method_override
        if not method_value:
            raise typer.BadParameter("method is missing; provide --method-override")
        method = parse_method(method_value)
        model_id = row.model_id or model_override
        if not model_id:
            raise typer.BadParameter("model_id is missing; provide --model-override")
        predictions.append(
            PredictionRecord(
                run_id=f"external-{method}-{index}",
                method=method,
                model_id=model_id,
                scenario_id=scenario.id,
                family=scenario.family,
                counterfactual_group=scenario.counterfactual_group,
                target=scenario.target_z.model_dump(),
                target_intensity=scenario.target_intensity,
                temperature=row.temperature,
                sample_index=index if row.sample_index is None else row.sample_index,
                prompt_variant=row.prompt_variant,
                style_family=scenario.style_family,
                prompt=scenario.situation,
                response=row.response,
                latency_ms=0.0,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_models(str(output), predictions)
    manifest_payload = {
        "schema_version": "prediction-v1",
        "raw_path": str(raw),
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "scenarios_path": str(scenarios),
        "scenarios_sha256": hashlib.sha256(scenarios.read_bytes()).hexdigest(),
        "source_manifest": str(source_manifest) if source_manifest else None,
        "source_manifest_payload": (
            json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest else None
        ),
        "n_predictions": len(predictions),
    }
    output.with_name(f"{output.stem}.manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(f"normalized {len(predictions)} official rows -> {output}")


if __name__ == "__main__":
    app()
