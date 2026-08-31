#!/usr/bin/env -S uv run --script
"""Create a balanced orthogonal Trait x Style evaluation manifest."""
from __future__ import annotations

from pathlib import Path
import typer
from persona_rl.schema import TargetTraits, parse_jsonl, write_jsonl

app = typer.Typer(no_args_is_help=True)
STYLES = ("neutral", "warm/polite", "blunt/direct", "formal", "terse", "conversational")

@app.command()
def main(input_path: Path, output_path: Path = Path("data/raw/trait_style_matrix.jsonl"), split: str = "test") -> None:
    rows = [r for r in parse_jsonl(str(input_path)) if r.split == split and r.hidden_task.target_trait]
    out = []
    for row in rows:
        trait = row.hidden_task.target_trait
        for level in (-1, 1):
            target = TargetTraits(**{name: (level if name == trait else 0) for name in TargetTraits.model_fields})
            for style in STYLES:
                out.append(row.model_copy(update={
                    "id": f"{row.id}:trait{level:+d}:style_{style.replace('/', '_')}" ,
                    "target_z": target,
                    "counterfactual_group": f"{row.counterfactual_group}:trait_style",
                    "style_family": style,
                }))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(output_path), out)
    typer.echo(f"wrote {len(out)} cells ({len(rows)} scenarios x 2 traits x {len(STYLES)} styles) to {output_path}")

if __name__ == "__main__":
    app()
