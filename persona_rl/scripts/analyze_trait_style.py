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
    reversals = []
    for (method, _base, style), values in pairs.items():
        if 1 in values and -1 in values:
            delta = values[1] - values[-1]; effects[(method, style)].append(delta)
            if delta < 0:
                reversals.append({"method": method, "style": style, "delta": delta, "high": values[1], "low": values[-1]})
    coefficients = {}
    methods = sorted({r["method"] for r in records})
    for method in methods:
        by_style = {style: sum(values) / len(values) for (candidate, style), values in effects.items() if candidate == method}
        values = list(by_style.values())
        center = sum(values) / len(values) if values else None
        coefficients[method] = {
            "trait_effect_by_style": by_style,
            "mean_trait_effect": center,
            "min_trait_effect": min(values) if values else None,
            "max_trait_effect": max(values) if values else None,
            "effect_range": max(values) - min(values) if values else None,
            "style_interaction_contrast": ({style: value - center for style, value in by_style.items()} if center is not None else {}),
            "n_cells": len([r for r in records if r["method"] == method]),
        }
    (output_dir / "trait_style_coefficients.json").write_text(json.dumps(coefficients, indent=2, ensure_ascii=False), encoding="utf-8")
    model = {method: {"formula": "Y ~ trait * style + (1|family) + (1|scenario)",
                      "estimator": "paired cell means; scenario/family are the pairing clusters",
                      "fixed_effects": value} for method, value in coefficients.items()}
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        frame = pd.DataFrame([{
            "behavior_validity": row.behavior_validity,
            "trait_level": next((v for v in row.prediction.target.values() if v), 0),
            "style_family": row.prediction.style_family,
            "family": row.prediction.family,
            "scenario_id": row.prediction.scenario_id,
            "method": row.prediction.method,
        } for row in rows])
        for method in methods:
            subset = frame[frame["method"] == method]
            if len(subset) >= 12 and subset["style_family"].nunique() >= 2:
                fitted = smf.mixedlm(
                    "behavior_validity ~ trait_level * C(style_family)",
                    subset, groups=subset["family"],
                ).fit(reml=False, disp=False)
                model[method]["estimator"] = "statsmodels MixedLM; family random intercept"
                model[method]["mixedlm_params"] = {str(k): float(v) for k, v in fitted.params.items()}
                model[method]["mixedlm_pvalues"] = {str(k): float(v) for k, v in fitted.pvalues.items()}
    except Exception:
        model["_note"] = "Install pandas and statsmodels for MixedLM; paired contrasts remain available without them."
    (output_dir / "trait_style_model.json").write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "conflict_reversals.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in reversals), encoding="utf-8")
    # Dependency-free SVG heatmap keeps analysis runnable on CPU-only servers.
    styles = sorted({r["style"] for r in records}); methods = sorted({r["method"] for r in records})
    width, height, cell = 180 + 100 * len(styles), 70 + 32 * len(methods), 28
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><style>text{{font:12px sans-serif}}</style>']
    for j, style in enumerate(styles): svg.append(f'<text x="{170+j*100}" y="18">{style}</text>')
    for i, method in enumerate(methods):
        svg.append(f'<text x="4" y="{45+i*32}">{method}</text>')
        for j, style in enumerate(styles):
            value = coefficients.get(method, {}).get("trait_effect_by_style", {}).get(style)
            color = "#cccccc" if value is None else ("#2166ac" if value >= 0 else "#b2182b")
            svg.append(f'<rect x="{150+j*100}" y="{25+i*32}" width="90" height="24" fill="{color}"/><text fill="white" x="{155+j*100}" y="42">{value:.3f}</text>' if value is not None else f'<rect x="{150+j*100}" y="{25+i*32}" width="90" height="24" fill="{color}"/>')
    svg.append('</svg>'); (output_dir / "trait_style_heatmap.svg").write_text("".join(svg), encoding="utf-8")
    typer.echo(f"wrote {len(records)} cells and coefficients for {len(coefficients)} methods")

if __name__ == "__main__": app()
