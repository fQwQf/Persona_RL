#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pydantic>=2.8", "typer>=0.12"]
# ///
"""Convert downloaded external datasets into frozen Scenario records."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from typing import Any
import typer
from persona_rl.schema import HiddenTask, Scenario, TargetTraits, write_jsonl

app = typer.Typer(no_args_is_help=True)

def _first(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip(): return value.strip()
        if isinstance(value, list) and value: return "\n".join(str(x) for x in value if str(x).strip())
    return ""

def _adapt(name: str, row: dict[str, Any]) -> tuple[str, str, tuple[str, ...], str]:
    if name == "truthful_qa":
        prompt = _first(row, ("question",))
        gold = _first(row, ("best_answer", "correct_answers"))
        return prompt, gold, ("truthfulness", "factuality"), "en"
    if name == "empathetic_dialogues":
        prompt = _first(row, ("utterance", "context", "prompt"))
        gold = _first(row, ("response", "utterance"))
        return prompt, gold, ("empathy", "agreeableness"), "en"
    if name == "prosocial_dialog":
        prompt = _first(row, ("context", "dialogue", "utterance", "prompt"))
        gold = _first(row, ("response", "rots", "safety_label"))
        return prompt, gold, ("prosocial", "safety"), "en"
    if name == "persona_chat":
        prompt = _first(row, ("history", "context", "dialog", "utterances"))
        persona = _first(row, ("personality", "persona", "persona_b"))
        return f"Persona context:\n{persona}\nConversation:\n{prompt}", persona, ("persona_consistency",), "en"
    raise typer.BadParameter(f"unsupported adapter: {name}")

@app.command()
def main(name: str, raw: Path, output: Path, limit: int = 1000, seed: int = 7) -> None:
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    random.Random(seed).shuffle(rows)
    scenarios = []
    for index, row in enumerate(rows):
        situation, gold, tags, language = _adapt(name, row)
        if not situation: continue
        scenarios.append(Scenario(
            id=f"external:{name}:{index:07d}", family=f"external_{name}", split="test",
            target_z=TargetTraits(conscientiousness=0, agreeableness=0, honesty_humility=0),
            situation=situation, behavior_rubric=(gold,) if gold else (),
            capability_rubric=(gold,) if gold else (), evaluation_tags=(f"source:{name}", *tags),
            hidden_task=HiddenTask(type="structured_check", question=situation, gold_behavior=1),
            counterfactual_group=f"external:{name}:{index:07d}", language=language,
        ))
        if len(scenarios) >= limit: break
    if not scenarios: raise typer.BadParameter("adapter produced no usable scenarios; inspect raw schema")
    output.parent.mkdir(parents=True, exist_ok=True); write_jsonl(str(output), scenarios)
    manifest = {"schema_version":"external-eval-v1", "dataset":name, "raw_path":str(raw),
                "raw_sha256":hashlib.sha256(raw.read_bytes()).hexdigest(), "seed":seed,
                "n_scenarios":len(scenarios), "training_allowed":False}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    typer.echo(f"prepared {len(scenarios)} frozen scenarios -> {output}")

if __name__ == "__main__": app()
