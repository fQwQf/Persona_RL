#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Run comparable methods and emit predictions, scores, manifests, and reports."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
try:
    from datetime import UTC
except ImportError:  # Python 3.10 compatibility for shared research servers.
    from datetime import timezone
    UTC = timezone.utc
from pathlib import Path

import typer

from persona_rl.baselines import baseline
from persona_rl.inference import PROMPT_VARIANTS, InferenceConfig, InferenceEngine
from persona_rl.reporting import write_report
from persona_rl.results import (
    PredictionRecord,
    RunManifest,
    ScoreRecord,
    parse_method,
    write_models,
)
from persona_rl.schema import parse_jsonl
from persona_rl.scoring import score_prediction

app = typer.Typer(no_args_is_help=True)
DEFAULT_METHODS = "base,prompt_only,sft,direct_dpo,pc_dpo"


def _model_map(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in raw.split(","):
        if "=" in item:
            key, value = item.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")


def _local_fingerprint() -> str:
    """Hash local source files so a run remains auditable without a parent Git repository."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(
        (*root.joinpath("src").rglob("*.py"), *root.joinpath("scripts").rglob("*.py"))
    ):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@app.command()
def main(
    scenarios: Path,
    output_dir: Path = Path("artifacts/experiment"),
    methods: str = DEFAULT_METHODS,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    model_revision: str = "",
    model_map: str = "",
    backend: str = "dry_run",
    split: str = "test",
    samples: int = 3,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    variants: str = "canonical,paraphrase,minimal,formal,terse,conversational",
    scorer: str = "none",
    judge_model: str = "Qwen/Qwen2.5-72B-Instruct",
    allow_proxy_external: bool = False,
) -> None:
    """Run methods with paired scenarios and render a combined report."""
    if backend not in {"dry_run", "openai", "hf"}:
        raise typer.BadParameter("backend must be dry_run, openai, or hf")
    if scorer not in {"llm", "smoke", "none"}:
        raise typer.BadParameter("scorer must be llm, smoke, or none")
    if scorer == "llm" and not os.environ.get("OPENAI_API_KEY"):
        raise typer.BadParameter(
            "scorer=llm requires OPENAI_API_KEY; use scorer=none for predictions only"
        )
    if samples < 1:
        raise typer.BadParameter("samples must be positive")
    if not 0 <= temperature <= 2:
        raise typer.BadParameter("temperature must be between zero and two")
    prompt_variants = tuple(value.strip() for value in variants.split(",") if value.strip())
    if not prompt_variants or any(value not in PROMPT_VARIANTS for value in prompt_variants):
        raise typer.BadParameter(f"variants must be a comma-separated subset of {PROMPT_VARIANTS}")
    run_id = _run_id()
    local_fingerprint = _local_fingerprint()
    endpoint = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    key = os.environ.get("OPENAI_API_KEY", "")
    records = [record for record in parse_jsonl(str(scenarios)) if record.split == split]
    if not records:
        raise typer.BadParameter(f"no scenarios found for split: {split}")
    if backend == "openai" and not key:
        raise typer.BadParameter("OPENAI_API_KEY is required for the openai backend")
    models = _model_map(model_map)
    selected_methods = [value.strip() for value in methods.split(",") if value.strip()]
    if not selected_methods:
        raise typer.BadParameter("methods must contain at least one method")
    output_run_dir = output_dir / run_id
    output_run_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "experiment_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "experiment-v1",
                "run_id": run_id,
                "methods": selected_methods,
                "model": model,
                "model_map": models,
                "model_revision": model_revision,
                "backend": backend,
                "split": split,
                "samples": samples,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "variants": prompt_variants,
                "scenarios_path": str(scenarios),
                "scenarios_sha256": hashlib.sha256(scenarios.read_bytes()).hexdigest(),
                "local_source_fingerprint": local_fingerprint,
                "seed": 7,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    score_paths: list[Path] = []
    prediction_paths: list[Path] = []
    for method in selected_methods:
        try:
            method_name = parse_method(method)
        except ValueError as exc:
            raise typer.BadParameter(f"unknown method: {method}") from exc
        external_method = method_name in {
            "machine_mindset",
            "personality_vector",
            "persllm",
            "rolellm",
            "personagym",
            "representation_engineering",
            "big5_chat",
            "simpo",
        }
        if external_method and not allow_proxy_external:
            raise typer.BadParameter(
                f"{method} requires run_official_baseline.py + normalize_external.py; "
                "pass --allow-proxy-external only for smoke tests"
            )
        method_model = models.get(method, model)
        if backend == "hf" and method in {"sft", "direct_dpo", "pc_dpo"} and method not in models:
            raise typer.BadParameter(
                f"hf evaluation requires an explicit checkpoint in --model-map for {method}"
            )
        if (
            method
            in {
                "machine_mindset",
                "personality_vector",
                "persllm",
                "rolellm",
                "personagym",
                "representation_engineering",
                "big5_chat",
                "simpo",
            }
            and method not in models
        ):
            try:
                method_model = baseline(method).default_model
            except ValueError:
                method_model = model
        method_dir = output_run_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        manifest = RunManifest(
            run_id=run_id,
            method=method_name,
            model_id=method_model,
            source_repo="persona_rl"
            if method in {"base", "prompt_only", "sft", "direct_dpo", "kto", "pc_dpo"}
            else baseline(method).repo,
            source_commit=local_fingerprint
            if method in {"base", "prompt_only", "sft", "direct_dpo", "kto", "pc_dpo"}
            else baseline(method).commit,
            command=" ".join(os.sys.argv),
            seed=7,
            config={
                "backend": backend,
                "split": split,
                "samples": str(samples),
                "variants": ",".join(prompt_variants),
                "rubric_version": "rubric-v1",
                "model_revision": model_revision,
            },
        )
        (method_dir / "run_manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        config = InferenceConfig(
            method_name,
            method_model,
            backend,
            endpoint,
            key,
            temperature,
            max_new_tokens,
            "You are a research evaluation assistant.",
            model_revision,
        )
        engine = InferenceEngine(config)
        predictions: list[PredictionRecord] = []
        for record in records:
            for sample_index in range(samples):
                for variant in prompt_variants:
                    prediction = engine.generate(record, sample_index, variant)
                    predictions.append(prediction.model_copy(update={"run_id": run_id}))
        prediction_path = method_dir / "predictions.jsonl"
        write_models(str(prediction_path), predictions)
        prediction_paths.append(prediction_path)
        score_path = method_dir / "scores.jsonl"
        if scorer == "smoke":
            scenario_map = {record.id: record for record in records}
            scores: list[ScoreRecord] = [
                score_prediction(prediction, scenario_map[prediction.scenario_id])
                for prediction in predictions
            ]
            write_models(str(score_path), scores)
            score_paths.append(score_path)
    combined_predictions = output_run_dir / "predictions.jsonl"
    with combined_predictions.open("w", encoding="utf-8") as handle:
        for path in prediction_paths:
            handle.write(path.read_text(encoding="utf-8"))
    if scorer == "llm":
        combined_scores = output_run_dir / "scores.jsonl"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("llm_score_outputs.py")),
                str(combined_predictions),
                str(scenarios),
                "--output",
                str(combined_scores),
                "--model",
                judge_model,
            ],
            check=True,
        )
        scored = read_models(str(combined_scores), ScoreRecord)
        for method in selected_methods:
            method_scores = [score for score in scored if score.prediction.method == method]
            score_path = output_run_dir / method / "scores.jsonl"
            write_models(str(score_path), method_scores)
            score_paths.append(score_path)
    combined = output_run_dir / "scores.jsonl"
    combined.parent.mkdir(parents=True, exist_ok=True)
    if scorer == "smoke":
        with combined.open("w", encoding="utf-8") as handle:
            for path in score_paths:
                handle.write(path.read_text(encoding="utf-8"))
        write_report(str(combined), str(output_run_dir / "report"))
    elif scorer == "llm":
        write_report(str(combined), str(output_run_dir / "report"))
    validation = {
        "ok": True,
        "run_dir": str(output_run_dir),
        "methods": selected_methods,
        "score_files": [str(path) for path in score_paths],
        "score_count": sum(path.read_text(encoding="utf-8").count("\n") for path in score_paths),
        "errors": [],
        "note": "Run-level emission check; use validate_artifacts.py for full CI validation.",
    }
    (output_run_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(
        json.dumps(
            {
                "run_id": run_id,
                "methods": selected_methods,
                "report": str(output_run_dir / "report/report.html") if scorer == "smoke" else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    app()
