#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///

"""Summarize interactive audit decisions without turning them into training labels."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer

from persona_rl.results import AuditRecord, read_models

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(audit: Path, output: Path = Path("artifacts/report/audit_summary.json")) -> None:
    """Write decision counts, reviewer counts, and mean quality score."""
    records = read_models(str(audit), AuditRecord)
    if not records:
        raise typer.BadParameter("audit JSONL is empty")
    decisions = Counter(record.decision for record in records)
    reviewers = Counter(record.reviewer for record in records)
    summary = {
        "n": len(records),
        "decision_counts": dict(sorted(decisions.items())),
        "reviewer_counts": dict(sorted(reviewers.items())),
        "mean_quality_score": sum(record.score for record in records) / len(records),
        "uncertain_rate": decisions.get("uncertain", 0) / len(records),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(
        "\n".join(
            [
                "# Human audit summary",
                "",
                f"Audited items: **{summary['n']}**",
                f"Mean quality score: **{summary['mean_quality_score']:.2f}/5**",
                f"Uncertain rate: **{summary['uncertain_rate']:.1%}**",
                "",
                "## Decisions",
                "",
                "| decision | count |",
                "|---|---:|",
                *[f"| {key} | {value} |" for key, value in sorted(decisions.items())],
                "",
                "## Reviewers",
                "",
                "| reviewer | count |",
                "|---|---:|",
                *[f"| {key} | {value} |" for key, value in sorted(reviewers.items())],
                "",
            ]
        ),
        encoding="utf-8",
    )
    typer.echo(f"wrote audit summary to {output} and {markdown}")


if __name__ == "__main__":
    app()
