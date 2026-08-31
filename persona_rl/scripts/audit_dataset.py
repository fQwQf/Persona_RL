#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///
"""Audit external or generated JSONL for duplicates, language balance, and provenance."""
from __future__ import annotations
import hashlib, json, re
from collections import Counter
from pathlib import Path
import typer

app = typer.Typer(no_args_is_help=True)

def _lang(text: str) -> str:
    return "zh" if len(re.findall(r"[\u4e00-\u9fff]", text)) >= 2 else "en"

@app.command()
def main(input_path: Path, output: Path = Path("artifacts/data_audit.json"), limit: int = 0) -> None:
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit > 0: rows = rows[:limit]
    ids = [str(row.get("id", row.get("scenario_id", ""))) for row in rows]
    texts = []
    for row in rows:
        text = " ".join(str(row.get(k, "")) for k in ("situation", "question", "context", "utterance", "text", "prompt"))
        texts.append(" ".join(text.lower().split()))
    duplicates = len(texts) - len(set(texts))
    payload = {"path": str(input_path), "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
               "n_rows": len(rows), "n_unique_ids": len(set(ids)), "duplicate_text_rows": duplicates,
               "language_counts": dict(Counter(_lang(text) for text in texts)),
               "empty_text_rows": sum(not text.strip() for text in texts)}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False))

if __name__ == "__main__": app()
