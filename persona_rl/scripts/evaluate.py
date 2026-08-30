#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Aggregate JSONL judge outputs into reproducible metric summaries."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from persona_rl.metrics import bootstrap_ci
from persona_rl.results import ScoreRecord, read_models

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(input_path: Path, output_path: Path = Path("artifacts/metrics.json")) -> None:
    """Compute means and grouped counts from evaluator records."""
    raw_rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not raw_rows:
        raise typer.BadParameter("input JSONL is empty")
    summary: dict[str, object]
    try:
        records = read_models(str(input_path), ScoreRecord)
    except ValueError:
        records = []
    if records:
        buckets: dict[str, list[float]] = {}
        for record in records:
            for metric in (
                "trait_fidelity",
                "behavior_validity",
                "truthfulness",
                "safety",
                "sycophancy",
            ):
                buckets.setdefault(f"{record.prediction.method}.{metric}", []).append(
                    float(getattr(record, metric))
                )
        summary = {
            "n": len(records),
            "means": {key: sum(values) / len(values) for key, values in buckets.items()},
            "confidence_intervals": {
                key: dict(zip(("low", "high"), bootstrap_ci(values), strict=True))
                for key, values in buckets.items()
            },
        }
    else:
        numeric: dict[str, list[float]] = {}
        for row in raw_rows:
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric.setdefault(key, []).append(float(value))
        summary = {
            "n": len(raw_rows),
            "means": {key: sum(values) / len(values) for key, values in numeric.items() if values},
            "confidence_intervals": {
                key: dict(zip(("low", "high"), bootstrap_ci(values), strict=True))
                for key, values in numeric.items()
                if values
            },
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
