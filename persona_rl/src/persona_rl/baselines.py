"""Pinned provenance for public baselines and benchmark-only repositories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """Public source and execution classification for a baseline."""

    name: str
    label: str
    repo: str
    commit: str
    local_dir: str
    paper: str
    license: str
    execution: str
    default_model: str


BASELINES: Final[tuple[BaselineSpec, ...]] = (
    BaselineSpec(
        "machine_mindset",
        "Machine Mindset MBTI DPO",
        "PKU-YuanGroup/Machine-Mindset",
        "5ee14c144fc59579a6c77e553acb2b6b56a54d7d",
        "external/Machine-Mindset",
        "https://arxiv.org/abs/2312.12999",
        "Apache-2.0",
        "released_checkpoint",
        "FarReelAILab/Machine_Mindset_en_ISTJ",
    ),
    BaselineSpec(
        "persllm",
        "PersLLM automatic DPO",
        "Ellenzzn/PersLLM",
        "f0b58b1896e716267f27d823c84b0511a0d0647d",
        "external/PersLLM",
        "https://arxiv.org/abs/2407.12393",
        "repository-license",
        "official_training",
        "miniCPM-mc-format",
    ),
    BaselineSpec(
        "personality_vector",
        "Geometry of Personality activation steering",
        "gunmayhanda/The-Geometry-of-Personality",
        "47628a9982c4532d19b16d7aa4c66228d0f304ad",
        "external/Geometry-of-Personality",
        "https://arxiv.org/abs/2407.17491",
        "repository-license",
        "official_notebooks",
        "google/gemma-2-2b-it",
    ),
    BaselineSpec(
        "rolellm",
        "RoleLLM RoCIT/RoleBench",
        "InteractiveNLP-Team/RoleLLM-public",
        "131a157c9962f46d36a29bcec6962b5acfa7644a",
        "external/RoleLLM-public",
        "https://arxiv.org/abs/2310.00746",
        "repository-license",
        "benchmark_only",
        "meta-llama/Llama-2-7b-chat-hf",
    ),
    BaselineSpec(
        "personagym",
        "PersonaGym dynamic evaluation",
        "vsamuel2003/PersonaGym",
        "536f705e610289c38d84e083fcea6ea5093d3a25",
        "external/PersonaGym",
        "https://arxiv.org/abs/2407.18416",
        "repository-license",
        "benchmark_only",
        "gpt-4o-mini",
    ),
    BaselineSpec(
        "representation_engineering",
        "Representation Engineering",
        "andyzoujm/representation-engineering",
        "5455d8a375d5fb1cb191f9ebcd089b7c21e9a31e",
        "external/representation-engineering",
        "https://arxiv.org/abs/2310.01405",
        "MIT",
        "official_notebooks",
        "google/gemma-2-2b-it",
    ),
    BaselineSpec(
        "big5_chat",
        "BIG5-CHAT human-grounded Big Five training data",
        "wenkai-li/Big5-Chat",
        "fb50e6a7af599c0cdb86df4e18a4d1bb89c28da8",
        "external/BIG5-CHAT",
        "https://arxiv.org/abs/2410.16491",
        "repository-license",
        "dataset_only",
        "wenkai-li/big5_chat",
    ),
    BaselineSpec(
        "simpo",
        "SimPO reference-free preference optimization",
        "princeton-nlp/SimPO",
        "1b3e8f3528a23bce3da514a2dce8ea7490d4bc75",
        "external/SimPO",
        "https://arxiv.org/abs/2405.14734",
        "MIT",
        "official_training",
        "princeton-nlp/gemma-2-9b-it-SimPO",
    ),
)


def baseline(name: str) -> BaselineSpec:
    """Return a pinned baseline specification or fail at the CLI boundary."""
    for spec in BASELINES:
        if spec.name == name:
            return spec
    raise ValueError(f"unknown baseline: {name}")
