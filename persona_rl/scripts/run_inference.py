#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Generate common-schema predictions for a method and a scenario split."""

from __future__ import annotations

import os
from pathlib import Path

import typer

from persona_rl.inference import PROMPT_VARIANTS, InferenceConfig, InferenceEngine
from persona_rl.results import MethodName, PredictionRecord, write_models
from persona_rl.schema import parse_jsonl

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    scenarios: Path,
    output: Path = Path("artifacts/predictions.jsonl"),
    method: MethodName = "base",
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    model_revision: str = "",
    backend: str = "dry_run",
    split: str = "test",
    samples: int = 3,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    variants: str = "canonical",
    rank: int | None = None,
    world_size: int | None = None,
) -> None:
    """Run inference with dry-run, OpenAI-compatible, or Hugging Face backend."""
    if backend not in {"dry_run", "openai", "hf"}:
        raise typer.BadParameter("backend must be dry_run, openai, or hf")
    if samples < 1:
        raise typer.BadParameter("samples must be positive")
    prompt_variants = tuple(value.strip() for value in variants.split(",") if value.strip())
    if not prompt_variants or any(value not in PROMPT_VARIANTS for value in prompt_variants):
        raise typer.BadParameter(f"variants must be a comma-separated subset of {PROMPT_VARIANTS}")
    rank_value = int(os.environ.get("RANK", "0")) if rank is None else rank
    world_value = int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else world_size
    if world_value < 1 or rank_value < 0 or rank_value >= world_value:
        raise typer.BadParameter("rank must be in [0, world_size) and world_size must be positive")
    records = [record for record in parse_jsonl(str(scenarios)) if record.split == split]
    records = [record for index, record in enumerate(records) if index % world_value == rank_value]
    endpoint = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    key = os.environ.get("OPENAI_API_KEY", "")
    config = InferenceConfig(
        method,
        model,
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
                predictions.append(engine.generate(record, sample_index, variant))
    actual_output = (
        output.with_name(f"{output.stem}.rank{rank_value}{output.suffix}")
        if world_value > 1
        else output
    )
    actual_output.parent.mkdir(parents=True, exist_ok=True)
    write_models(str(actual_output), predictions)
    typer.echo(
        f"rank {rank_value}/{world_value}: wrote {len(predictions)} predictions to {actual_output}"
    )


if __name__ == "__main__":
    app()
