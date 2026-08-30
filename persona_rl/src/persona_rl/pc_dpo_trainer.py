"""Differentiable constraint-weighted DPO trainer."""

from __future__ import annotations

from typing import Any


def _sequence_logprob(model: Any, input_ids: Any, attention_mask: Any) -> Any:
    """Return length-normalized causal log probability for a token batch."""
    import torch

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].to(logits.dtype)
    token_logprobs = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def build_trainer(base: type[Any]) -> type[Any]:
    """Create a TRL-compatible trainer with differentiable constraint penalties."""

    class ConstraintDPOTrainer(base):
        """DPO trainer adding criterion, invariance, truth and uncertainty penalties."""

        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
            import torch

            metadata = {
                key: inputs.pop(key, None)
                for key in ("criterion_weight", "invariance_weight", "truth_weight", "safety_weight", "uncertainty")
            }
            base_loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)
            criterion = metadata["criterion_weight"]
            invariance = metadata["invariance_weight"]
            truth = metadata["truth_weight"]
            safety = metadata["safety_weight"]
            uncertainty = metadata["uncertainty"]
            if any(value is None for value in (criterion, invariance, truth, safety, uncertainty)):
                return (base_loss, {}) if return_outputs else base_loss
            chosen = _sequence_logprob(model, inputs["chosen_input_ids"], inputs["chosen_attention_mask"])
            rejected = _sequence_logprob(model, inputs["rejected_input_ids"], inputs["rejected_attention_mask"])
            separation = torch.sigmoid(chosen - rejected)
            penalty = (
                (1 - criterion) * (1 - separation)
                + (1 - invariance) * (1 - separation)
                + (1 - truth) * torch.relu(-chosen)
                + (1 - safety) * torch.relu(-chosen)
                + uncertainty * separation
            ).mean()
            reference = getattr(self, "ref_model", None)
            if reference is not None:
                with torch.no_grad():
                    reference_chosen = _sequence_logprob(
                        reference, inputs["chosen_input_ids"], inputs["chosen_attention_mask"]
                    )
                penalty = penalty + 0.05 * (chosen - reference_chosen).pow(2).mean()
            loss = base_loss + 0.25 * penalty
            return (loss, {"base_loss": base_loss.detach(), "constraint_penalty": penalty.detach()}) if return_outputs else loss

    return ConstraintDPOTrainer
