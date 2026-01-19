from typing import Tuple

import torch

from modeling.generic.indexing.candidate_index import TopKModule


class MIPSTopKModule(TopKModule):

    def __init__(
        self,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> None:
        """
        Args:
            item_embeddings: (1, X, D)
            item_ids: (1, X,)
        """
        super().__init__()

        self._item_embeddings: torch.Tensor = item_embeddings
        self._item_ids: torch.Tensor = item_ids


class MIPSBruteForceTopK(MIPSTopKModule):

    def __init__(
        self,
        item_embeddings: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> None:
        super().__init__(
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
        del self._item_embeddings
        self._item_embeddings_t: torch.Tensor = item_embeddings.permute(2, 1, 0).squeeze(2)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        k: int,
        sorted_results: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query_embeddings: (B, ...). Implementation-specific.
            k: int. final top-k to return.
            sorted_results: bool. whether to sort final top-k results or not.

        Returns:
            Tuple of (top_k_scores x float, top_k_ids x int), both of shape (B, K,)
        """
        all_logits = torch.mm(query_embeddings, self._item_embeddings_t)
        top_k_logits, top_k_indices = torch.topk(
            all_logits, dim=1, k=k, sorted=sorted_results, largest=True,
        )  # (B, k,)
        return top_k_logits, self._item_ids.squeeze(0)[top_k_indices]
    
def get_top_k_module(top_k_method: str, model: torch.nn.Module, item_embeddings: torch.Tensor,
                     item_ids: torch.Tensor) :
    if top_k_method == "MIPSBruteForceTopK":
        top_k_module = MIPSBruteForceTopK(
            item_embeddings=item_embeddings,
            item_ids=item_ids,
        )
    else:
        raise ValueError(f"Invalid top-k method {top_k_method}")
    return top_k_module