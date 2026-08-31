#!/usr/bin/env -S uv run --script
"""Pre-download a reproducible model using Hugging Face or ModelScope."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import typer
app = typer.Typer(no_args_is_help=True)

@app.command()
def main(model: str, output: Path = Path("models/base"), source: str = "huggingface", revision: str = "") -> None:
    if source not in {"huggingface", "modelscope"}:
        raise typer.BadParameter("source must be huggingface or modelscope")
    output.mkdir(parents=True, exist_ok=True)
    if source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise typer.BadParameter("install modelscope first: uv pip install modelscope") from exc
        path = Path(snapshot_download(model, revision=revision or None, cache_dir=str(output)))
    else:
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo_id=model, revision=revision or None, local_dir=str(output / model.replace('/', '__'))))
    files = sorted(p for p in path.rglob('*') if p.is_file())
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path)).encode()); digest.update(file.read_bytes())
    manifest = {"model": model, "source": source, "revision": revision or None, "path": str(path), "n_files": len(files), "tree_sha256": digest.hexdigest()}
    (output / "model.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    typer.echo(json.dumps(manifest, indent=2))
if __name__ == "__main__": app()
