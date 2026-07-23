"""Text-only vLLM adapter for Gemma4 Unified checkpoints."""

from collections.abc import Iterable

import torch

from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM


class Gemma4UnifiedTextOnlyForCausalLM(Gemma4ForCausalLM):
    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        text_weights = (
            (name, weight)
            for name, weight in weights
            if "vision_embedder." not in name
        )
        return super().load_weights(text_weights)
