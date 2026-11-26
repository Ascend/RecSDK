"""Paper-structured text/image item fusion for OxygenREC Semantic IDs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MultimodalFusionOutput:
    item_embedding: Tensor
    query_tokens: Tensor


class ResidualQFormerLayer(nn.Module):
    def __init__(self, hidden_size: int, attention_heads: int, feedforward_size: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.memory_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, attention_heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size), nn.GELU(),
            nn.Linear(feedforward_size, hidden_size),
        )

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        attended, _ = self.cross_attention(
            self.query_norm(queries), self.memory_norm(memory), self.memory_norm(memory),
            need_weights=False,
        )
        queries = queries + attended
        return queries + self.ffn(self.ffn_norm(queries))


class MultimodalItemEncoder(nn.Module):
    """Fuse precomputed text/image tokens through residual Q-Former layers.

    Qwen3/CLIP remain replaceable upstream encoders. This module reproduces the
    disclosed fusion boundary without pretending RetailRocket contains raw
    titles or images.
    """

    def __init__(
        self, *, text_size: int, image_size: int, hidden_size: int = 256,
        query_tokens: int = 8, qformer_layers: int = 2,
        attention_heads: int = 8, output_size: int = 256,
    ) -> None:
        super().__init__()
        if min(text_size, image_size, hidden_size, query_tokens, qformer_layers, output_size) < 1:
            raise ValueError("all multimodal dimensions and layer counts must be positive")
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self.text_projection = nn.Linear(text_size, hidden_size)
        self.image_projection = nn.Linear(image_size, hidden_size)
        self.modality_embedding = nn.Embedding(2, hidden_size)
        self.learned_queries = nn.Parameter(torch.empty(query_tokens, hidden_size))
        self.layers = nn.ModuleList(
            ResidualQFormerLayer(hidden_size, attention_heads, hidden_size * 4)
            for _ in range(qformer_layers)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, output_size)
        )
        nn.init.normal_(self.learned_queries, std=0.02)

    def forward(self, text_tokens: Tensor, image_tokens: Tensor) -> MultimodalFusionOutput:
        if text_tokens.ndim != 3 or image_tokens.ndim != 3:
            raise ValueError("text_tokens and image_tokens must be [batch, tokens, features]")
        if text_tokens.shape[0] != image_tokens.shape[0]:
            raise ValueError("text and image batch sizes must match")
        text = self.text_projection(text_tokens) + self.modality_embedding.weight[0]
        image = self.image_projection(image_tokens) + self.modality_embedding.weight[1]
        memory = torch.cat((text, image), dim=1)
        queries = self.learned_queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        # Every layer refines the preceding query state through residual updates.
        for layer in self.layers:
            queries = layer(queries, memory)
        item_embedding = self.output(queries.mean(dim=1))
        return MultimodalFusionOutput(item_embedding=item_embedding, query_tokens=queries)
