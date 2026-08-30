#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12", "rich>=13.7"]
# ///

"""Interactive low-volume quality audit for generated scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from persona_rl.schema import parse_jsonl

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    input_path: Path, output_path: Path = Path("data/processed/audit.jsonl"), limit: int = 100
) -> None:
    """Review scenarios with 1-5 scores and write an audit trail."""
    records = parse_jsonl(str(input_path))[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            typer.echo(f"\n[{record.id}] {record.family}\n{record.situation}")
            typer.echo(f"Target: {record.target_z.model_dump()}")
            valid = typer.confirm(
                "Is the rubric machine-checkable and non-ambiguous?", default=True
            )
            score = typer.prompt("Quality score 1-5", default=5, type=int)
            if not 1 <= score <= 5:
                raise typer.BadParameter("quality score must be between 1 and 5")
            note = typer.prompt("Note", default="")
            result = {"id": record.id, "valid": valid, "score": score, "note": note}
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    typer.echo(f"audited {len(records)} records -> {output_path}")


if __name__ == "__main__":
    app()
