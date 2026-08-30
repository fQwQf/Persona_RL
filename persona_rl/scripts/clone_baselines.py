#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Clone pinned public baseline repositories through GitHub CLI."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import typer

from persona_rl.baselines import BASELINES, BaselineSpec

app = typer.Typer(no_args_is_help=True)


def _clone(root: Path, spec: BaselineSpec) -> str:
    destination = root / spec.local_dir
    if not (destination / ".git").is_dir():
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                ["gh", "repo", "clone", spec.repo, str(destination), "--", "--depth", "1"],
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("GitHub CLI `gh` is required to clone public baselines") from exc
    actual = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if actual != spec.commit:
        dirty = subprocess.run(
            ["git", "-C", str(destination), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if dirty:
            raise RuntimeError(f"cannot move dirty baseline checkout: {destination}")
        subprocess.run(
            ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", spec.commit],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", spec.commit], check=True
        )
        actual = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    if actual != spec.commit:
        raise RuntimeError(
            f"{spec.name} resolved to {actual}, expected pinned commit {spec.commit}"
        )
    return actual


@app.command()
def main(root: Path = Path("."), manifest: Path = Path("configs/baseline_manifest.json")) -> None:
    """Clone sources and write the expected provenance manifest."""
    root = root.resolve()
    manifest_path = manifest if manifest.is_absolute() else root / manifest
    verified = []
    for spec in BASELINES:
        resolved_commit = _clone(root, spec)
        verified.append(
            {
                **asdict(spec),
                "resolved_commit": resolved_commit,
                "clone_command": ["gh", "repo", "clone", spec.repo],
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(verified, indent=2), encoding="utf-8")
    typer.echo(f"verified {len(BASELINES)} pinned baseline sources")


if __name__ == "__main__":
    app()
