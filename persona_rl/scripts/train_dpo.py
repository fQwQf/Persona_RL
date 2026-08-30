#!/usr/bin/env python

"""Launch Direct-DPO or PC-DPO with TRL when training extras are installed."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def main(
    dataset: Path,
    model: str = "Qwen/Qwen2.5-7B-Instruct",
    model_revision: str = "",
    output: Path = Path("artifacts/checkpoints/pc_dpo"),
    method: str = "pc-dpo",
    seed: int = 7,
    epochs: float = 1.0,
    batch_size: int = 1,
    grad_accumulation: int = 8,
    learning_rate: float = 5e-6,
    max_length: int = 2048,
    gradient_checkpointing: bool = True,
    use_lora: bool = True,
    qlora: bool = True,
    bf16: bool = True,
    resume_from_checkpoint: Path | None = None,
) -> None:
    """Run training through TRL; use torchrun for multi-GPU execution."""
    try:
        import torch
        from datasets import Dataset, load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise typer.BadParameter("install the train extra: uv sync --extra train") from exc
    if method not in {"direct-dpo", "pc-dpo"}:
        raise typer.BadParameter("method must be direct-dpo or pc-dpo")
    if not dataset.exists():
        raise typer.BadParameter(f"dataset does not exist: {dataset}")
    if qlora and not torch.cuda.is_available():
        raise typer.BadParameter("QLoRA requires CUDA; pass --no-qlora for a CPU smoke run")
    if qlora and not use_lora:
        raise typer.BadParameter("QLoRA requires LoRA; pass --no-qlora when disabling LoRA")
    data = load_dataset("json", data_files=str(dataset), split="train")
    if method == "pc-dpo":
        from persona_rl.constraints import reward_from_judge

        rows = data.to_list()
        weighted = []
        for row in rows:
            raw_reward = row.get("pc_reward", {})
            reward = reward_from_judge(raw_reward if isinstance(raw_reward, dict) else {})
            copies = max(0, min(2, round(reward.weight())))
            weighted.extend(
                [
                    {
                        **row,
                        "criterion_weight": reward.criterion,
                        "invariance_weight": reward.invariance,
                        "truth_weight": reward.truth,
                        "safety_weight": reward.safety,
                        "uncertainty": reward.uncertainty,
                    }
                ]
                * copies
            )
        if not weighted:
            raise typer.BadParameter("PC-DPO reward constraints rejected every training pair")
        data = Dataset.from_list(weighted)
    columns = ["prompt", "chosen", "rejected"]
    if method == "pc-dpo":
        columns.extend(["criterion_weight", "invariance_weight", "truth_weight", "safety_weight", "uncertainty"])
    data = data.select_columns(columns)
    revision = model_revision or None
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cuda = torch.cuda.is_available()
    if cuda:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    use_bf16 = bool(bf16 and cuda and torch.cuda.is_bf16_supported())
    model_kwargs: dict[str, object] = {}
    if qlora:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": torch.cuda.current_device()}
    if revision is not None:
        model_kwargs["revision"] = revision
    policy = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
    if qlora and use_lora:
        policy = prepare_model_for_kbit_training(policy)
    policy.config.use_cache = False
    args = DPOConfig(
        output_dir=str(output),
        seed=seed,
        num_train_epochs=epochs,
        bf16=use_bf16,
        fp16=bool(cuda and not use_bf16),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        max_length=max_length,
        gradient_checkpointing=gradient_checkpointing,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        report_to=[],
    )
    peft_config = None
    if use_lora:
        target_modules = sorted(
            {
                name.rsplit(".", 1)[-1]
                for name, _module in policy.named_modules()
                if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj"))
            }
        )
        if not target_modules:
            raise typer.BadParameter("could not find q/k/v/o projection modules for LoRA")
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
    if method == "pc-dpo":
        from persona_rl.pc_dpo_trainer import build_trainer

        trainer_type = build_trainer(DPOTrainer)
    else:
        trainer_type = DPOTrainer
    trainer = trainer_type(
        model=policy,
        args=args,
        train_dataset=data,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if method == "pc-dpo":
        trainer.args.remove_unused_columns = False
    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
    )
    trainer.save_model(str(output))
    (output / "training_config.json").write_text(
        json.dumps(
            {
                "method": method,
                "objective": "reward-weighted-constraint-dpo" if method == "pc-dpo" else "standard-dpo",
                "constraint_heads": ["trait", "criterion", "invariance", "truth", "safety", "uncertainty"] if method == "pc-dpo" else [],
                "model": model,
                "model_revision": model_revision,
                "dataset": str(dataset),
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accumulation": grad_accumulation,
                "learning_rate": learning_rate,
                "max_length": max_length,
                "use_lora": use_lora,
                "qlora": qlora,
                "bf16": use_bf16,
                "local_rank": os.environ.get("LOCAL_RANK", "0"),
                "resume_from_checkpoint": str(resume_from_checkpoint)
                if resume_from_checkpoint
                else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"saved {method} checkpoint to {output}")


if __name__ == "__main__":
    app()
