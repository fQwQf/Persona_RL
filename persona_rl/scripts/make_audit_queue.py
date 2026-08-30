#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Create a deterministic, stratified human-audit sample from score records."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import typer

from persona_rl.results import ScoreRecord, read_models, write_models

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    scores: Path,
    output: Path = Path("artifacts/report/audit_queue.jsonl"),
    fraction: float = 0.2,
    seed: int = 7,
) -> None:
    """Sample at least one case per method/family and prioritize flagged cases."""
    if not 0 < fraction <= 1:
        raise typer.BadParameter("fraction must be in (0, 1]")
    records = read_models(str(scores), ScoreRecord)
    groups: dict[tuple[str, str], list[ScoreRecord]] = defaultdict(list)
    for record in records:
        groups[(record.prediction.method, record.prediction.family)].append(record)
    rng = random.Random(seed)
    selected: list[ScoreRecord] = []
    for group_records in groups.values():
        n = max(1, round(len(group_records) * fraction))
        flagged = [
            record for record in group_records if record.rule_flags or record.judge_confidence < 0.8
        ]
        priority = flagged[:n]
        remainder = [record for record in group_records if record not in priority]
        selected.extend(priority + rng.sample(remainder, min(n - len(priority), len(remainder))))
    selected.sort(
        key=lambda record: (
            record.prediction.method,
            record.prediction.family,
            record.prediction.scenario_id,
            record.prediction.prompt_variant,
            record.prediction.sample_index,
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_models(str(output), selected)
    output.with_name(f"{output.stem}.manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "audit-queue-v1",
                "scores_path": str(scores),
                "seed": seed,
                "fraction": fraction,
                "n_input": len(records),
                "n_selected": len(selected),
                "strata": len(groups),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    typer.echo(f"sampled {len(selected)} of {len(records)} scores -> {output}")


if __name__ == "__main__":
    app()
