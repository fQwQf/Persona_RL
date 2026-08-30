#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Plan or execute the original repository command for a pinned baseline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import typer

from persona_rl.baselines import baseline

app = typer.Typer(no_args_is_help=True)


def _command(method: str, root: Path) -> tuple[tuple[str, ...], Path]:
    commands: dict[str, tuple[tuple[str, ...], Path]] = {
        "persllm": (("bash", "src/train.sh"), root / "external/PersLLM/training_codes"),
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
        ),
        "personagym": (
            ("python", "run.py", "--benchmark", "benchmark-v1"),
            root / "external/PersonaGym/code",
        ),
        "rolellm": (
            ("python", "-c", "print('RoleLLM is benchmark-only in the pinned public repository')"),
            root,
        ),
        "machine_mindset": (("python", "cli_inference.py"), root / "external/Machine-Mindset"),
        "big5_chat": (
            (
                "python",
                "-c",
                "print('BIG5-CHAT is a dataset-only source; use its release as an SFT "
                "data baseline')",
            ),
            root,
        ),
        "simpo": (
            (
                "bash",
                "-lc",
                "printf '%s\\n' 'SimPO official training: use training_configs/*.yaml "
                "and scripts/run_simpo.py'",
            ),
            root / "external/SimPO",
        ),
    }
    if method not in commands:
        raise ValueError(f"method has no official command: {method}")
    return commands[method]


@app.command()
def main(
    method: str,
    root: Path = Path("."),
    output_dir: Path = Path("artifacts/official_runs"),
    execute: bool = False,
) -> None:
    """Record source provenance and optionally execute the original command."""
    root = root.resolve()
    spec = baseline(method)
    source = root / spec.local_dir
    actual = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if actual != spec.commit:
        raise typer.BadParameter(f"source commit mismatch: {actual} != {spec.commit}")
    command, command_cwd = _command(method, root)
    destination = output_dir / method
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": method,
        "repo": spec.repo,
        "commit": actual,
        "paper": spec.paper,
        "license": spec.license,
        "command": list(command),
        "cwd": str(command_cwd),
        "execution": spec.execution,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if execute:
        with (destination / "stdout.log").open("w", encoding="utf-8") as stdout:
            subprocess.run(
                command, cwd=command_cwd, stdout=stdout, stderr=subprocess.STDOUT, check=True
            )
    typer.echo(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    app()
