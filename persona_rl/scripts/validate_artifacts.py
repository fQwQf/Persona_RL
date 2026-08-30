#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Validate one experiment directory before publishing its metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from persona_rl.results import PredictionRecord, RunManifest, ScoreRecord, read_models

app = typer.Typer(no_args_is_help=True)


def _keys(records: list[PredictionRecord]) -> set[tuple[str, str, int]]:
    return {(record.scenario_id, record.prompt_variant, record.sample_index) for record in records}


@app.command()
def main(
    run_dir: Path,
    output: Path | None = None,
    expected_methods: str = "",
    expected_variants: str = "",
) -> None:
    """Check provenance, row alignment, uniqueness, and report artifacts."""
    errors: list[str] = []
    experiment_manifest_path = run_dir / "experiment_manifest.json"
    experiment_manifest: dict[str, object] = {}
    if not experiment_manifest_path.exists():
        errors.append(f"missing {experiment_manifest_path}")
    else:
        try:
            experiment_manifest = json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"invalid experiment manifest: {exc}")
    method_dirs = (
        sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name != "report")
        if run_dir.exists()
        else []
    )
    expected = {value.strip() for value in expected_methods.split(",") if value.strip()}
    found = {path.name for path in method_dirs}
    if expected and found != expected:
        errors.append(
            f"method directories differ: expected={sorted(expected)} found={sorted(found)}"
        )
    method_prediction_keys: dict[str, set[tuple[str, str, int]]] = {}
    method_score_count = 0
    rubric_versions: set[str] = set()
    for method_dir in method_dirs:
        manifest_path = method_dir / "run_manifest.json"
        prediction_path = method_dir / "predictions.jsonl"
        score_path = method_dir / "scores.jsonl"
        for required in (manifest_path, prediction_path, score_path):
            if not required.exists():
                errors.append(f"missing {required}")
        if not manifest_path.exists() or not prediction_path.exists() or not score_path.exists():
            continue
        try:
            manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            predictions = read_models(str(prediction_path), PredictionRecord)
            scores = read_models(str(score_path), ScoreRecord)
        except ValueError as exc:
            errors.append(f"invalid records in {method_dir}: {exc}")
            continue
        if manifest.method != method_dir.name:
            errors.append(f"manifest method mismatch in {method_dir}")
        if experiment_manifest.get("run_id") != manifest.run_id:
            errors.append(f"run_id mismatch in {manifest_path}")
        if not manifest.source_commit:
            errors.append(f"missing source_commit in {manifest_path}")
        prediction_keys = _keys(predictions)
        if len(prediction_keys) != len(predictions):
            errors.append(f"duplicate prediction keys in {prediction_path}")
        score_keys = _keys([score.prediction for score in scores])
        rubric_versions.update(score.rubric_version for score in scores)
        if prediction_keys != score_keys:
            errors.append(f"prediction/score key mismatch in {method_dir}")
        if len(predictions) != len(scores):
            errors.append(f"prediction/score count mismatch in {method_dir}")
        method_prediction_keys[method_dir.name] = prediction_keys
        method_score_count += len(scores)
        if expected_variants:
            allowed = {value.strip() for value in expected_variants.split(",") if value.strip()}
            actual = {record.prompt_variant for record in predictions}
            if actual != allowed:
                errors.append(
                    f"variants differ in {prediction_path}: expected={sorted(allowed)} "
                    f"found={sorted(actual)}"
                )
    combined_path = run_dir / "scores.jsonl"
    if not combined_path.exists():
        errors.append(f"missing {combined_path}")
    else:
        try:
            combined = read_models(str(combined_path), ScoreRecord)
            if len(combined) != method_score_count:
                errors.append("combined score count differs from method score files")
        except ValueError as exc:
            errors.append(f"invalid combined scores: {exc}")
    report_dir = run_dir / "report"
    for required in (
        "report.md",
        "report.html",
        "report_manifest.json",
        "summary.csv",
        "summary.json",
        "review_queue.jsonl",
    ):
        if not (report_dir / required).exists():
            errors.append(f"missing report artifact: {report_dir / required}")
    report_manifest_path = report_dir / "report_manifest.json"
    if report_manifest_path.exists() and combined_path.exists():
        try:
            report_manifest = json.loads(report_manifest_path.read_text(encoding="utf-8"))
            expected_hash = hashlib.sha256(combined_path.read_bytes()).hexdigest()
            if report_manifest.get("input_sha256") != expected_hash:
                errors.append("report manifest input hash differs from combined scores")
        except json.JSONDecodeError as exc:
            errors.append(f"invalid report manifest: {exc}")
    if len(rubric_versions) > 1:
        errors.append(f"multiple rubric versions found: {sorted(rubric_versions)}")
    result = {
        "ok": not errors,
        "run_dir": str(run_dir),
        "methods": sorted(found),
        "prediction_counts": {method: len(keys) for method, keys in method_prediction_keys.items()},
        "score_count": method_score_count,
        "errors": errors,
    }
    destination = output or (run_dir / "validation.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
