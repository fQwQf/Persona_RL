#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Generate candidate responses through an OpenAI-compatible chat endpoint."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import typer

from persona_rl.schema import parse_jsonl

app = typer.Typer(no_args_is_help=True)


def _request(endpoint: str, key: str, model: str, prompt: str, temperature: float) -> str:
    body = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return str(payload["choices"][0]["message"]["content"])


@app.command()
def main(
    input_path: Path,
    output_path: Path = Path("data/processed/candidate_pairs.jsonl"),
    model: str = "gpt-4o-mini",
    limit: int = 0,
    candidates_per_scenario: int = 2,
    resume: bool = True,
    split: str = "train",
) -> None:
    """Expand scenarios into paired candidates for later judge filtering."""
    key = os.environ.get("OPENAI_API_KEY", "")
    endpoint = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    if not key:
        raise typer.BadParameter("set OPENAI_API_KEY or use a local OpenAI-compatible endpoint")
    records = parse_jsonl(str(input_path))
    if candidates_per_scenario < 2:
        raise typer.BadParameter("candidates_per_scenario must be at least 2")
    if split not in {"train", "validation", "test", "audit"}:
        raise typer.BadParameter("split must be train, validation, test, or audit")
    split_records = [record for record in records if record.split == split]
    selected = split_records if limit <= 0 else split_records[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    manifest_path = output_path.with_name(f"{output_path.stem}.manifest.json")
    if resume and output_path.exists() and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(
            previous.get(field) != value
            for field, value in {
                "input_sha256": input_sha256,
                "model": model,
                "split": split,
                "candidates_per_scenario": candidates_per_scenario,
            }.items()
        ):
            raise typer.BadParameter("candidate output provenance changed; use a new output path")
    existing = set()
    if resume and output_path.exists():
        existing = {
            json.loads(line)["id"]
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    mode = "a" if resume else "w"
    generated_count = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        for record in selected:
            if record.id in existing:
                continue
            prompt = json.dumps(
                {
                    "situation": record.situation,
                    "history": record.history,
                    "target_z": record.target_z.model_dump(),
                    "hard_constraints": record.hard_constraints,
                    "rubric": record.behavior_rubric,
                    "forbidden": record.forbidden_behaviors,
                    "capability_rubric": record.capability_rubric,
                },
                ensure_ascii=False,
            )
            candidates = [
                _request(
                    endpoint,
                    key,
                    model,
                    (
                        "Generate one response that follows every rubric and avoids all forbidden "
                        "behavior. Return only the response.\n"
                        if index % 2 == 0
                        else "Generate one plausible response that fails exactly one rubric or "
                        "hard constraint while sounding helpful. Return only the response.\n"
                    )
                    + prompt,
                    min(1.0, 0.7 + (index * 0.1)),
                )
                for index in range(candidates_per_scenario)
            ]
            handle.write(
                json.dumps(
                    {
                        "id": record.id,
                        "prompt": record.situation,
                        "family": record.family,
                        "split": record.split,
                        "target_z": record.target_z.model_dump(),
                        "counterfactual_group": record.counterfactual_group,
                        "candidate_a": candidates[0],
                        "candidate_b": candidates[1],
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            generated_count += 1
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "candidate-expansion-v1",
                "input_path": str(input_path),
                "input_sha256": input_sha256,
                "output_path": str(output_path),
                "model": model,
                "split": split,
                "candidates_per_scenario": candidates_per_scenario,
                "n_rows": len(existing) + generated_count,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    typer.echo(f"generated paired candidates for {generated_count} scenarios -> {output_path}")


if __name__ == "__main__":
    app()
