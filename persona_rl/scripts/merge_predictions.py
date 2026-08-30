#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Merge rank-sharded prediction JSONL files without silently dropping duplicates."""

from __future__ import annotations

from pathlib import Path

import typer

from persona_rl.results import PredictionRecord, read_models, write_models

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    input_dir: Path,
    output: Path = Path("artifacts/predictions.jsonl"),
    pattern: str = "*.rank*.jsonl",
) -> None:
    """Merge per-rank outputs and validate one unique row per condition."""
    inputs = sorted(input_dir.glob(pattern))
    if not inputs:
        raise typer.BadParameter(f"no shard files matched {pattern!r} in {input_dir}")
    records: list[PredictionRecord] = []
    seen: set[tuple[str, str, str, int]] = set()
    for path in inputs:
        for record in read_models(str(path), PredictionRecord):
            key = (
                record.method,
                record.scenario_id,
                record.prompt_variant,
                record.sample_index,
            )
            if key in seen:
                raise typer.BadParameter(f"duplicate prediction key {key} in {path}")
            seen.add(key)
            records.append(record)
    records.sort(
        key=lambda record: (
            record.method,
            record.scenario_id,
            record.prompt_variant,
            record.sample_index,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_models(str(output), records)
    typer.echo(f"merged {len(inputs)} shards and {len(records)} predictions -> {output}")


if __name__ == "__main__":
    app()
