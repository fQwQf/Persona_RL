#!/usr/bin/env -S uv run --script
"""Aggregate Trait x Style scores and estimate simple interaction contrasts."""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import typer
from persona_rl.results import ScoreRecord, read_models

app = typer.Typer(no_args_is_help=True)

@app.command()
def main(scores: Path, output_dir: Path = Path("artifacts/trait_style")) -> None:
    rows = read_models(str(scores), ScoreRecord)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        p = row.prediction
        trait = next((k for k, v in p.target.items() if v), "none")
        level = int(p.target.get(trait, 0)) if trait != "none" else 0
        groups[(p.method, trait, level, p.prompt_variant, p.family)].append(row)
    cells = defaultdict(list)
    for (method, trait, level, style, family), values in groups.items():
        cells[(method, trait, level, style)].append(sum(v.behavior_validity for v in values) / len(values))
    records = []
    for key, vals in sorted(cells.items()):
        method, trait, level, style = key
        records.append({"method": method, "trait": trait, "level": level, "style": style,
                        "n_families": len(vals), "behavior_validity": sum(vals)/len(vals)})
    with (output_dir / "trait_style_cells.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys() if records else ["method"])
        writer.writeheader(); writer.writerows(records)
    coeff = {}
    for method in sorted({r["method"] for r in records}):
        by_style = defaultdict(dict)
        for r in records:
            if r["method"] == method: by_style[(r["trait"], r["style"])][r["level"]] = r["behavior_validity"]
        effects = [v[1] - v[-1] for v in by_style.values() if 1 in v and -1 in v]
        coeff[method] = {"mean_trait_effect": sum(effects)/len(effects) if effects else None,
                         "min_trait_effect": min(effects) if effects else None,
                         "n_cells": len([r for r in records if r["method"] == method])}
    (output_dir / "trait_style_coefficients.json").write_text(json.dumps(coeff, indent=2), encoding="utf-8")
    typer.echo(f"wrote {len(records)} cells and coefficients for {len(coeff)} methods to {output_dir}")

if __name__ == "__main__": app()
