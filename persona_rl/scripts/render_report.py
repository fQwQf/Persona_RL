#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Render readable Markdown, HTML, CSV, and audit queue artifacts."""

from __future__ import annotations

from pathlib import Path

import typer

from persona_rl.reporting import write_report

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(scores: Path, output_dir: Path = Path("artifacts/report")) -> None:
    """Render a score JSONL file into a standalone report directory."""
    write_report(str(scores), str(output_dir))
    typer.echo(f"report written to {output_dir}/report.md and report.html")


if __name__ == "__main__":
    app()
