"""Common inference backends and method prompt adapters."""

from __future__ import annotations

import json
import os
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field

from .results import MethodName, PredictionRecord
from .schema import Scenario

BackendName = Literal["dry_run", "openai", "hf"]
PromptVariant = Literal["canonical", "paraphrase", "minimal", "formal", "terse", "conversational"]
PROMPT_VARIANTS: tuple[PromptVariant, ...] = (
    "canonical", "paraphrase", "minimal", "formal", "terse", "conversational"
)


class ChoiceMessage(BaseModel):
    """Validated chat response content from an OpenAI-compatible endpoint."""

    model_config = ConfigDict(frozen=True)
    content: str = ""


class Choice(BaseModel):
    """Validated completion choice."""

    model_config = ConfigDict(frozen=True)
    message: ChoiceMessage


class ChatResponse(BaseModel):
    """Validated completion response envelope."""

    model_config = ConfigDict(frozen=True)
    choices: tuple[Choice, ...] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Runtime configuration shared by local and remote backends."""

    method: MethodName
    model_id: str
    backend: BackendName
    endpoint: str
    api_key: str
    temperature: float
    max_new_tokens: int
    system_prompt: str
    model_revision: str = ""


PROFILES: Final[dict[MethodName, str]] = {
    "base": "Answer as a helpful assistant without a specified personality.",
    "prompt_only": (
        "Express the requested target traits through observable behavior, not trait words."
    ),
    "sft": "Follow the target behavior profile and satisfy the stated constraints.",
    "direct_dpo": "Follow the target behavior profile and satisfy the stated constraints.",
    "kto": "Follow the target behavior profile and satisfy the stated constraints.",
    "personality_vector": "Use the target behavior profile while preserving factuality and safety.",
    "machine_mindset": (
        "Respond with a stable personality style while preserving factuality and safety."
    ),
    "persllm": "Respond as a consistent personified agent while preserving factuality and safety.",
    "rolellm": "Use the role-conditioned behavior while preserving factuality and safety.",
    "personagym": (
        "Follow the supplied persona evaluation context while preserving factuality and safety."
    ),
    "representation_engineering": (
        "Use the controlled personality representation while preserving factuality and safety."
    ),
    "big5_chat": (
        "Use the trained Big Five behavior profile while preserving factuality and safety."
    ),
    "simpo": (
        "Use the reference-free preference-optimized behavior while preserving "
        "factuality and safety."
    ),
    "pc_dpo": "Follow target behavior while prioritizing truthfulness and safety constraints.",
}


def build_prompt(
    record: Scenario, method: MethodName, system_prompt: str, variant: PromptVariant = "canonical"
) -> str:
    """Build a comparable prompt without exposing rubric labels to evaluation text."""
    target = record.target_z.model_dump()
    history = "\n".join(record.history)
    if variant == "canonical":
        return (
            f"{system_prompt}\nMethod adapter: Follow the supplied target behavior code.\n"
            f"Target behavior code: {target}; intensity={record.target_intensity:.2f}\nSituation: {record.situation}\n"
            f"Conversation history:\n{history}\nRespond to the user."
        )
    if variant == "paraphrase":
        return (
            f"{system_prompt}\nUse the supplied target behavior code.\n"
            f"The target code is {target} at intensity {record.target_intensity:.2f}. First read the situation below, then answer it.\n"
            f"Situation: {record.situation}\nPrior turns:\n{history}\n"
            "Give the final user-facing answer."
        )
    if variant == "minimal":
        return (
            f"{system_prompt}\nAdapter=target_behavior\nTarget={target}; intensity={record.target_intensity:.2f}\n"
            f"User situation={record.situation}\nHistory={history}\nAnswer:"
        )
    if variant == "formal":
        return f"{system_prompt}\nTarget={target}\nSituation={record.situation}\nProvide a formal answer."
    if variant == "terse":
        return f"{system_prompt}\nTarget={target}\nSituation={record.situation}\nAnswer briefly."
    if variant == "conversational":
        return f"{system_prompt}\nTarget={target}\nSituation={record.situation}\nReply conversationally."
    raise ValueError(f"unknown prompt variant: {variant}")


def _openai_completion(config: InferenceConfig, prompt: str) -> str:
    body = {
        "model": config.model_id,
        "temperature": config.temperature,
        "max_tokens": config.max_new_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        config.endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        parsed = ChatResponse.model_validate_json(response.read())
    return parsed.choices[0].message.content


def _make_hf_runner(config: InferenceConfig) -> Callable[[str], str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("HF backend requires uv sync --extra train") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    revision = config.model_revision or None
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter_config = Path(config.model_id) / "adapter_config.json"
    model_kwargs: dict[str, object] = {
        "torch_dtype": torch.bfloat16 if device == "cuda" else None,
    }
    if revision is not None:
        model_kwargs["revision"] = revision
    if adapter_config.exists():
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("loading a LoRA checkpoint requires the train extra") from exc
        adapter = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_model_id = str(adapter["base_model_name_or_path"])
        model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
        model = PeftModel.from_pretrained(model, config.model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    model = model.to(device)
    model.eval()

    def run(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=config.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        )

    return run


class InferenceEngine:
    """Reusable inference engine that loads a local model only once per process."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config
        self._hf_runner = _make_hf_runner(config) if config.backend == "hf" else None

    def _response(self, prompt: str, record: Scenario) -> str:
        match self.config.backend:
            case "dry_run":
                return f"{record.behavior_rubric[0]} 我会说明不确定性，并遵守约束。"
            case "openai":
                return _openai_completion(self.config, prompt)
            case "hf":
                if self._hf_runner is None:
                    raise RuntimeError("HF runner was not initialized")
                return self._hf_runner(prompt)
            case unreachable:
                assert_never(unreachable)

    def generate(
        self,
        record: Scenario,
        sample_index: int,
        variant: PromptVariant = "canonical",
        language: str = "zh",
    ) -> PredictionRecord:
        """Generate one response using the cached backend."""
        prompt = build_prompt(record, self.config.method, self.config.system_prompt, variant)
        started = time.perf_counter()
        response = self._response(prompt, record)
        elapsed = (time.perf_counter() - started) * 1000
        return PredictionRecord(
            run_id=f"{self.config.method}-{self.config.model_id}-{sample_index}",
            method=self.config.method,
            model_id=self.config.model_id,
            scenario_id=record.id,
            family=record.family,
            counterfactual_group=record.counterfactual_group,
            target=record.target_z.model_dump(),
            target_intensity=record.target_intensity,
            prompt_variant=variant,
            language=language,
            temperature=self.config.temperature,
            sample_index=sample_index,
            prompt=prompt,
            response=response,
            latency_ms=elapsed,
        )


def generate(
    config: InferenceConfig,
    record: Scenario,
    sample_index: int,
    variant: PromptVariant = "canonical",
    language: str = "zh",
    engine: InferenceEngine | None = None,
) -> PredictionRecord:
    """Generate one response and return the common prediction record."""
    selected = engine if engine is not None else InferenceEngine(config)
    return selected.generate(record, sample_index, variant, language)


def environment_endpoint() -> tuple[str, str]:
    """Read an OpenAI-compatible endpoint and key from the process environment."""
    return (
        os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        os.environ.get("OPENAI_API_KEY", ""),
    )
