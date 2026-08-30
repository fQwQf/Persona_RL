#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Unified wrapper for pinned external baselines and common-schema outputs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer

from persona_rl.baselines import baseline

app = typer.Typer(no_args_is_help=True)


def _official_command(method: str, root: Path) -> tuple[tuple[str, ...], Path, str]:
    """Return the original command, working directory, and execution caveat."""
    commands: dict[str, tuple[tuple[str, ...], Path, str]] = {
        "machine_mindset": (
            ("python", "cli_inference.py"),
            root / "external/Machine-Mindset",
            "interactive CLI; provide scripted stdin or use a released checkpoint",
        ),
        "persllm": (
            ("bash", "src/train.sh"),
            root / "external/PersLLM/training_codes",
            "requires ModelCenter-format checkpoint and original Harry Potter data",
        ),
        "personality_vector": (
            (
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "activation_vector_extraction.ipynb",
            ),
            root / "external/Geometry-of-Personality",
            "notebook uses its released Gemma-2-2B-IT vector workflow",
        ),
        "representation_engineering": (
            (
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "examples/honesty/honesty.ipynb",
            ),
            root / "external/representation-engineering",
            "official RepE honesty notebook; adapt its control hook to personality vectors",
        ),
        "rolellm": (
            (
                "python",
                "-c",
                "print('RoleLLM public checkout contains RoleBench assets "
                "but no runnable trainer')",
            ),
            root / "external/RoleLLM-public",
            "benchmark/data-only checkout",
        ),
        "personagym": (
            ("python", "run.py", "--benchmark", "benchmark-v1"),
            root / "external/PersonaGym/code",
            "requires external API keys and model endpoint",
        ),
        "big5_chat": (
            (
                "python",
                "-c",
                "print('BIG5-CHAT is a dataset/model source; use its official "
                "Hugging Face artifacts')",
            ),
            root,
            "dataset/model source; no standalone local checkpoint in checkout",
        ),
        "simpo": (
            (
                "bash",
                "-lc",
                "printf '%s\\n' 'Use external/SimPO/scripts/run_simpo.py with an "
                "official training_configs YAML'",
            ),
            root / "external/SimPO",
            "official SimPO trainer/configs; choose the YAML matching the base model",
        ),
    }
    if method not in commands:
        raise typer.BadParameter(f"no external wrapper for {method}")
    return commands[method]


@app.command()
def main(
    method: str,
    root: Path = Path("."),
    output_dir: Path = Path("artifacts/official_runs"),
    execute: bool = False,
    raw_output: Path | None = None,
    scenarios: Path | None = None,
    normalized_output: Path | None = None,
    model_override: str = "",
) -> None:
    """Record pinned provenance and optionally run the original method command."""
    root = root.resolve()
    spec = baseline(method)
    source = root / spec.local_dir
    actual = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if actual != spec.commit:
        raise typer.BadParameter(f"source commit mismatch: {actual} != {spec.commit}")
    command, command_cwd, caveat = _official_command(method, root)
    destination = output_dir / method
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "external-run-v1",
        "method": method,
        "repo": spec.repo,
        "commit": actual,
        "paper": spec.paper,
        "license": spec.license,
        "execution_class": spec.execution,
        "command": list(command),
        "cwd": str(command_cwd),
        "default_model": spec.default_model,
        "caveat": caveat,
        "executed": execute,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if execute:
        with (destination / "stdout.log").open("w", encoding="utf-8") as stdout:
            subprocess.run(
                command, cwd=command_cwd, stdout=stdout, stderr=subprocess.STDOUT, check=True
            )
    if raw_output is not None or normalized_output is not None:
        if raw_output is None or scenarios is None or normalized_output is None:
            raise typer.BadParameter(
                "raw_output, scenarios, and normalized_output must be provided together"
            )
        normalize_command = [
            "uv",
            "run",
            "scripts/normalize_external.py",
            str(raw_output),
            str(scenarios),
            "--output",
            str(normalized_output),
            "--method-override",
            method,
            "--model-override",
            model_override or spec.default_model,
            "--source-manifest",
            str(destination / "manifest.json"),
        ]
        subprocess.run(normalize_command, cwd=root, check=True)
        manifest["normalized_output"] = str(normalized_output)
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    typer.echo(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
