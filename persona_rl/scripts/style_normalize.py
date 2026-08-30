#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Rewrite outputs into a common neutral style without changing the selected action."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import typer

from persona_rl.results import PredictionRecord, read_models, write_models

app = typer.Typer(no_args_is_help=True)


def _rewrite(endpoint: str, key: str, model: str, response: str) -> str:
    prompt = (
        "Rewrite the answer into concise neutral professional prose. Preserve every factual "
        "claim, uncertainty statement, refusal, and requested action. Do not add or remove an "
        "action. Return only the rewritten answer.\n" + response
    )
    body = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as handle:
        return str(json.load(handle)["choices"][0]["message"]["content"])


@app.command()
def main(
    predictions: Path,
    output: Path = Path("artifacts/normalized_predictions.jsonl"),
    model: str = "Qwen/Qwen2.5-72B-Instruct",
) -> None:
    """Create a cached style-normalized prediction file for robustness analysis."""
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise typer.BadParameter("set OPENAI_API_KEY")
    endpoint = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    rows = read_models(str(predictions), PredictionRecord)
    normalized = [
        row.model_copy(
            update={
                "response": _rewrite(endpoint, key, model, row.response),
                "run_id": f"{row.run_id}:style_normalized",
                "prompt_variant": "style_normalized",
            }
        )
        for row in rows
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_models(str(output), normalized)
    output.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "style-normalization-v1",
                "input_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
                "model": model,
                "n": len(normalized),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"normalized {len(normalized)} responses -> {output}")


if __name__ == "__main__":
    app()
