#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Convert validated scenarios into DPO prompt/chosen/rejected JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from persona_rl.schema import Scenario, parse_jsonl

app = typer.Typer(no_args_is_help=True)


def _prompt(record: Scenario) -> str:
    traits = record.target_z.model_dump()
    return f"Target traits: {traits}\nSituation: {record.situation}\nRespond helpfully."


@app.command()
def main(input_path: Path, output_path: Path = Path("data/processed/preferences.jsonl")) -> None:
    """Build deterministic placeholder pairs for smoke tests and DPO wiring."""
    records = parse_jsonl(str(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if not record.behavior_rubric or not record.forbidden_behaviors:
                raise typer.BadParameter(
                    f"scenario {record.id} must define behavior_rubric and forbidden_behaviors"
                )
            chosen = f"{record.behavior_rubric[0]} 不伪造事实，并说明不确定性。"
            rejected = record.forbidden_behaviors[0]
            row = {
                "prompt": _prompt(record),
                "chosen": chosen,
                "rejected": rejected,
                "id": record.id,
                "scenario_id": record.id,
                "family": record.family,
                "split": record.split,
                "counterfactual_group": record.counterfactual_group,
                "target_z": record.target_z.model_dump(),
                "source": "deterministic_smoke_placeholder",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    typer.echo(f"wrote {len(records)} preference pairs to {output_path}")


if __name__ == "__main__":
    app()
