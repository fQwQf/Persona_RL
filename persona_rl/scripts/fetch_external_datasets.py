#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=2.20", "typer>=0.12"]
# ///
"""Download allow-listed Hugging Face datasets with immutable provenance."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import typer

app = typer.Typer(no_args_is_help=True)
REGISTRY = Path(__file__).resolve().parents[1] / "configs" / "external_datasets.json"

@app.command()
def main(name: str, output_dir: Path = Path("data/raw/external"), revision: str = "") -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if name not in registry:
        raise typer.BadParameter(f"unknown dataset {name}; choose from {', '.join(registry)}")
    spec = registry[name]
    if spec.get("source") == "github":
        raise typer.BadParameter("this entry is GitHub-backed; use gh repo clone and record the commit")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise typer.BadParameter("install training/data dependencies: uv sync --extra train") from exc
    kwargs = {"split": spec["split"]}
    if revision:
        kwargs["revision"] = revision
    if spec.get("config"):
        dataset = load_dataset(spec["hf_id"], spec["config"], **kwargs)
    else:
        dataset = load_dataset(spec["hf_id"], **kwargs)
    target = output_dir / name
    target.mkdir(parents=True, exist_ok=True)
    out = target / f"{spec['split']}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in dataset:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {"name": name, **spec, "revision": revision or None, "hf_endpoint": __import__("os").environ.get("HF_ENDPOINT"), "path": str(out), "sha256": digest, "n_rows": len(dataset)}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(json.dumps(manifest, indent=2, ensure_ascii=False))

if __name__ == "__main__": app()
