#!/usr/bin/env -S uv run --script
"""Aggregate Trait x Style cells and paired trait effects."""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path
import typer
from persona_rl.results import ScoreRecord, read_models

app = typer.Typer(no_args_is_help=True)

@app.command()
def main(scores: Path, output_dir: Path = Path("artifacts/trait_style")) -> None:
    rows = read_models(str(scores), ScoreRecord); output_dir.mkdir(parents=True, exist_ok=True)
    cells = defaultdict(list); pairs = defaultdict(dict)
    for row in rows:
        p = row.prediction
        active = [(k, v) for k, v in p.target.items() if v]
        trait, level = active[0] if active else ("none", 0)
        cells[(p.method, trait, level, p.style_family)].append(row.behavior_validity)
        base_id = p.scenario_id.split(":trait", 1)[0]
        pairs[(p.method, base_id, p.style_family)][level] = row.behavior_validity
    records = []
    for (method, trait, level, style), values in sorted(cells.items()):
        records.append({"method": method, "trait": trait, "level": level, "style": style,
                        "n": len(values), "behavior_validity": sum(values) / len(values)})
    fields = ["method", "trait", "level", "style", "n", "behavior_validity"]
    with (output_dir / "trait_style_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    effects = defaultdict(list)
    for (method, _base, style), values in pairs.items():
        if 1 in values and -1 in values: effects[(method, style)].append(values[1] - values[-1])
    coefficients = {}
    methods = sorted({r["method"] for r in records})
    for method in methods:
        by_style = {style: sum(values) / len(values) for (candidate, style), values in effects.items() if candidate == method}
        values = list(by_style.values())
        coefficients[method] = {
            "trait_effect_by_style": by_style,
            "mean_trait_effect": sum(values) / len(values) if values else None,
            "min_trait_effect": min(values) if values else None,
            "max_trait_effect": max(values) if values else None,
            "effect_range": max(values) - min(values) if values else None,
            "n_cells": len([r for r in records if r["method"] == method]),
        }
    (output_dir / "trait_style_coefficients.json").write_text(json.dumps(coefficients, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"wrote {len(records)} cells and coefficients for {len(coefficients)} methods")

if __name__ == "__main__": app()
