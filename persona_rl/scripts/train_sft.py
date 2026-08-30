#!/usr/bin/env python

"""Train the supervised LoRA baseline with the same model and token budget."""

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
    output: Path = Path("artifacts/checkpoints/sft"),
    seed: int = 7,
    epochs: float = 1.0,
    batch_size: int = 1,
    grad_accumulation: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 2048,
    gradient_checkpointing: bool = True,
    use_lora: bool = True,
    qlora: bool = True,
    bf16: bool = True,
    resume_from_checkpoint: Path | None = None,
) -> None:
    """Run TRL SFT on the generated chosen responses."""
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise typer.BadParameter("install the train extra: uv sync --extra train") from exc
    if not dataset.exists():
        raise typer.BadParameter(f"dataset does not exist: {dataset}")
    if qlora and not torch.cuda.is_available():
        raise typer.BadParameter("QLoRA requires CUDA; pass --no-qlora for a CPU smoke run")
    if qlora and not use_lora:
        raise typer.BadParameter("QLoRA requires LoRA; pass --no-qlora when disabling LoRA")
    data = load_dataset("json", data_files=str(dataset), split="train").select_columns(
        ["prompt", "chosen"]
    )
    data = data.map(lambda row: {"text": f"{row['prompt']}\n{row['chosen']}"})
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
    args = SFTConfig(
        output_dir=str(output),
        seed=seed,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accumulation,
        learning_rate=learning_rate,
        max_length=max_length,
        dataset_text_field="text",
        gradient_checkpointing=gradient_checkpointing,
        bf16=use_bf16,
        fp16=bool(cuda and not use_bf16),
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        report_to=[],
        remove_unused_columns=False,
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
    trainer = SFTTrainer(
        model=policy,
        args=args,
        train_dataset=data,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(
        resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None
    )
    trainer.save_model(str(output))
    (output / "training_config.json").write_text(
        json.dumps(
            {
                "method": "sft",
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
    typer.echo(f"saved sft checkpoint to {output}")


if __name__ == "__main__":
    app()
