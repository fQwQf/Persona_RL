#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Review low-confidence prediction records without changing experiment data."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from persona_rl.results import AuditRecord, ScoreRecord, read_models

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    queue: Path, output: Path = Path("artifacts/report/human_audit.jsonl"), limit: int = 100
) -> None:
    """Interactively label a sampled review queue and append an audit log."""
    records = read_models(str(queue), ScoreRecord)[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    reviewer = os.environ.get("PERSONA_RL_REVIEWER", "anonymous")
    audited = set()
    if output.exists():
        audited = {record.item_id for record in read_models(str(output), AuditRecord)}
    new_count = 0
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            prediction = record.prediction
            item_id = (
                f"{prediction.method}:{prediction.scenario_id}:"
                f"{prediction.sample_index}:{prediction.prompt_variant}"
            )
            if item_id in audited:
                continue
            typer.echo(f"\n[{item_id}] {prediction.method}/{prediction.family}")
            typer.echo(
                f"Target: {prediction.target}\nResponse:\n{prediction.response}\n"
                f"Flags: {record.rule_flags}"
            )
            decision = typer.prompt("Decision accept/reject/uncertain", default="uncertain")
            if decision not in {"accept", "reject", "uncertain"}:
                raise typer.BadParameter("decision must be accept, reject, or uncertain")
            score = typer.prompt("Quality score 1-5", default=3, type=int)
            note = typer.prompt("Note", default="")
            audit = AuditRecord(
                item_id=item_id, decision=decision, score=score, note=note, reviewer=reviewer
            )
            handle.write(json.dumps(audit.model_dump(mode="json"), ensure_ascii=False) + "\n")
            new_count += 1
    typer.echo(f"audited {new_count} new records ({len(records)} considered) -> {output}")


if __name__ == "__main__":
    app()
