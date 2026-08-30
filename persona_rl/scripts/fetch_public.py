#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["typer>=0.12"]
# ///

"""Fetch public benchmark files by URL with a recorded manifest."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(url: str, output: Path, sha256: str = "") -> None:
    """Download a file and optionally verify its SHA-256 checksum."""
    request = urllib.request.Request(url, headers={"User-Agent": "persona-rl/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if sha256 and digest != sha256:
        raise typer.BadParameter(f"checksum mismatch: expected {sha256}, got {digest}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    manifest = output.with_suffix(output.suffix + ".manifest.json")
    manifest.write_text(json.dumps({"url": url, "sha256": digest}, indent=2), encoding="utf-8")
    typer.echo(f"downloaded {len(payload)} bytes to {output}")


if __name__ == "__main__":
    app()
