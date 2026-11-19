import os
import torch
import torch.distributed as dist
from modeling.generic.sequential_v2.features import SequentialFeatures
from modeling.generic.utils.constants import MAX_K
from typing import Callable, Dict, List, Optional, Set
from modeling.generic.indexing.candidate_index import CandidateIndex, TopKModule
from dataclasses import dataclass
 
@dataclass
class EvalState:
    all_item_ids: Set[int]
    candidate_index: CandidateIndex
    top_k_module: TopKModule
 
def eval_metrics_v2_from_tensors(
        eval_state: EvalState,
        model,
        model_input,
        seq_features: SequentialFeatures,
        target_ids,
        min_positive_rating: int = 1,
        target_ratings: Optional[torch.Tensor] = None,  # [B, 1]
        epoch: Optional[str] = None,
        filter_invalid_ids: bool = True,
        user_max_batch_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
) -> Dict[str, List[float]]:
    """
    Args:
        eval_negatives_ids: Optional[Tensor]. If not present, defaults to eval over
            the entire corpus (`num_items`) excluding all the items that users have
            seen in the past (historical_ids, target_ids). This is consistent with
            papers like SASRec and TDM but may not be fair in practice as retrieval
            modules don't have access to read state during the initial fetch stage.
        filter_invalid_ids: bool. If true, filters seen ids by default.
    Returns:
        keyed metric -> list of values for each example.
    """
    B, _ = target_ids.shape
    device = target_ids.device

    shared_input_embeddings = model.encode(
        past_lengths=seq_features.past_lengths,
        past_ids=seq_features.past_ids,
        past_embeddings=None,
        past_payloads=seq_features.past_payloads,
        model_inputs=model_input,
    )
    if dtype is not None:
        shared_input_embeddings = shared_input_embeddings.to(dtype)
    return shared_input_embeddings

def get_eval_state(
        model,
        all_item_ids: List[int],
        negatives_sampler,
        top_k_module_fn: Callable[[torch.Tensor, torch.Tensor], TopKModule],
        device: torch.device,
        float_dtype: Optional[torch.dtype] = None,
) -> EvalState:
    eval_negatives_ids = torch.as_tensor(all_item_ids).to(device).unsqueeze(0)
    eval_negative_embeddings = negatives_sampler.embedding_module.get_all_item_id_only_embeddings(eval_negatives_ids)
    eval_negative_embeddings = negatives_sampler.normalize_embeddings(eval_negative_embeddings)
    if float_dtype is not None:
        eval_negative_embeddings = eval_negative_embeddings.to(float_dtype)
    candidates = CandidateIndex(
        ids=eval_negatives_ids,
        embeddings=eval_negative_embeddings,
    )
    return EvalState(
        all_item_ids=set(all_item_ids),
        candidate_index=candidates,
        top_k_module=top_k_module_fn(eval_negative_embeddings, eval_negatives_ids)
    )

def _avg(x: torch.Tensor, world_size: int) -> float:
    if x:
        _sum_and_numel = torch.tensor([x.sum(), x.numel()], dtype=torch.float32, device=x.device)
        if world_size > 1:
            dist.all_reduce(_sum_and_numel, op=dist.ReduceOp.SUM)
        results = _sum_and_numel[0] / _sum_and_numel[1]
    else:
        results = 0
    return results