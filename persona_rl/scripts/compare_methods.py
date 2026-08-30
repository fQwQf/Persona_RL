#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Create paired method deltas from a combined score JSONL file."""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import typer

from persona_rl.metrics import bootstrap_ci
from persona_rl.reporting import METRICS
from persona_rl.results import ScoreRecord, read_models

app = typer.Typer(no_args_is_help=True)


def _average(rows: list[dict[str, str | float]], metric: str) -> float:
    return sum(float(row[metric]) for row in rows) / len(rows)


def _scenario_means(rows: list[dict[str, str | float]], metric: str) -> list[float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["scenario_id"]), []).append(float(row[metric]))
    return [sum(values) / len(values) for values in grouped.values()]


@app.command()
def main(scores: Path, output: Path = Path("artifacts/report/pairwise.csv")) -> None:
    """Write per-scenario deltas and a readable Markdown companion."""
    records = read_models(str(scores), ScoreRecord)
    grouped: dict[tuple[str, str, int], dict[str, ScoreRecord]] = {}
    for record in records:
        key = (
            record.prediction.scenario_id,
            record.prediction.prompt_variant,
            record.prediction.sample_index,
        )
        grouped.setdefault(key, {})[record.prediction.method] = record
    methods = sorted({record.prediction.method for record in records})
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []
    for left, right in combinations(methods, 2):
        for key, values in grouped.items():
            if left not in values or right not in values:
                continue
            row: dict[str, str | float] = {"scenario_id": key[0], "left": left, "right": right}
            for metric in METRICS:
                row[metric] = float(getattr(values[right], metric)) - float(
                    getattr(values[left], metric)
                )
            rows.append(row)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario_id", "left", "right", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    markdown = output.with_suffix(".md")
    lines = [
        "# Pairwise method deltas",
        "",
        f"Rows: {len(rows)}",
        "",
        "| left | right | mean behavior delta | mean safety delta | mean sycophancy delta |",
        "|---|---|---:|---:|---:|",
    ]
    for left, right in combinations(methods, 2):
        pair = [row for row in rows if row["left"] == left and row["right"] == right]
        if not pair:
            continue
        behavior_values = _scenario_means(pair, "behavior_validity")
        safety_values = _scenario_means(pair, "safety")
        sycophancy_values = _scenario_means(pair, "sycophancy")
        behavior_low, behavior_high = bootstrap_ci(behavior_values)
        safety_low, safety_high = bootstrap_ci(safety_values)
        sycophancy_low, sycophancy_high = bootstrap_ci(sycophancy_values)
        behavior_mean = sum(behavior_values) / len(behavior_values)
        safety_mean = sum(safety_values) / len(safety_values)
        sycophancy_mean = sum(sycophancy_values) / len(sycophancy_values)
        lines.append(
            f"| {left} | {right} | {behavior_mean:.3f} "
            f"[{behavior_low:.3f}, {behavior_high:.3f}] | "
            f"{safety_mean:.3f} [{safety_low:.3f}, {safety_high:.3f}] | "
            f"{sycophancy_mean:.3f} [{sycophancy_low:.3f}, {sycophancy_high:.3f}] |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"wrote {len(rows)} pairwise rows to {output} and {markdown}")


if __name__ == "__main__":
    app()
