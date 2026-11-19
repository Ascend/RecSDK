import torch
import torch.nn.functional as F
import logging
from typing import List, Tuple, Dict
from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const

class NegativesSampler(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)


    def normalize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return self._maybe_l2_norm(x)
    
    def load_embedding_module(self, embedding_module: BaseModel):
        self.embedding_module = embedding_module

    def _maybe_l2_norm(self, x: torch.Tensor) -> torch.Tensor:
        if self._l2_norm:
            squared_sum = torch.sum(x**2, dim=-1, keepdim=True)
            x = x / torch.clamp(
                torch.sqrt(torch.clamp(squared_sum, 0.0) + 1e-10),
                min=self._l2_norm_eps,
            )
        return x

@ModelRegistry.register(req_hp=True)
class InBatchNegativesSampler(NegativesSampler):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp["feature_conf"]
        self._dedup_embeddings: bool = feat_conf.get('dedup_embeddings', False)
        self._num_to_sample: int = model_cfg[Const.HP].get("num_to_sample")
        self._l2_norm = True
        self._l2_norm_eps = model_cfg[Const.HP].get("l2_norm_eps", 1e-6)

    def debug_str(self) -> str:
        sampling_debug_str = (
            f"in-batch{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        )
        if self._dedup_embeddings:
            sampling_debug_str += "-dedup"
        return sampling_debug_str

    def process_batches(self, ids, presences):
        """
        Args:
           ids: (N') or (B, N) x int64
           presences: (N') or (B, N) x bool
           embeddings: (N', D) or (B, N, D) x float
        """
        if self._dedup_embeddings:
            valid_ids = ids[presences]
            unique_ids, unique_ids_inverse_indices = torch.unique(
                input=valid_ids, sorted=False, return_inverse=True
            )
            device = unique_ids.device
            unique_embedding_offsets = torch.empty(
                (unique_ids.numel(),),
                dtype=torch.int64,
                device=device,
            )
            unique_embedding_offsets[unique_ids_inverse_indices] = torch.arange(
                valid_ids.numel(), dtype=torch.int64, device=device
            )
            unique_embeddings = embeddings[presences][unique_embedding_offsets, :]
            self._cached_embeddings = self._maybe_l2_norm(  # pyre-ignore [16]
                unique_embeddings
            )
            self._cached_ids = unique_ids  # pyre-ignore [16]
        else:
            self._cached_embeddings =  self.normalize_embeddings(self.embedding_module.get_all_item_id_only_embeddings(ids)[presences])
            self._cached_ids = ids[presences]
    
    def forward(
        self,
        model_inputs,
        positive_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings,).
        """
        past_ids = model_inputs['past_ids']
        self.process_batches(ids=past_ids,presences=past_ids!=0)
        X = self._cached_ids.size(0)
        sampled_offsets = torch.randint(
            low=0,
            high=X,
            size=positive_ids.size() + (self._num_to_sample,),
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        return (
            self._cached_ids[sampled_offsets],  # pyre-ignore [29]
            self._cached_embeddings[sampled_offsets],  # pyre-ignore [29]
        )
    
    def get_all_ids_and_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._cached_ids, self._cached_embeddings  # pyre-ignore [7]
    
@ModelRegistry.register(req_hp=True)
class LocalNegativesSampler(NegativesSampler):
    """
    精排模型中用于next item predicition的负采样器
    "hp": {"l2_norm": float,
            "l2_norm_eps": float,
            "num_to_sample": int
            }
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp["feature_conf"]
        self._l2_norm = model_cfg[Const.HP].get("l2_norm")
        self._l2_norm_eps = model_cfg[Const.HP].get("l2_norm_eps")
        self._num_to_sample: int = model_cfg[Const.HP].get("num_to_sample")
        itemid_column = feat_conf["raw_itemid_column"]
        max_item_id = feat_conf["item_feature_columns"][itemid_column]["feature_count"]
        all_item_ids = [x + 1 for x in range(max_item_id)]
        self._feat_conf = feat_conf 
        self._num_items = len(all_item_ids)
        self.register_buffer('_all_item_ids', torch.tensor(all_item_ids))

    def set_all_item_ids(self, valid_item_ids) -> None:
        self._num_items = len(valid_item_ids)
        self.register_buffer('_all_item_ids', torch.tensor(valid_item_ids))
    
    def get_all_ids_and_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._cached_ids, self._cached_embeddings

    def debug_str(self) -> str:
        sampling_debug_str = f"local{f'-l2-eps{self._l2_norm_eps}' if self._l2_norm else ''}"
        return sampling_debug_str


    def process_batches(self, sampled_ids, model_inputs):
        output_shape = sampled_ids.size()
        device = model_inputs['past_ids'].device
        model_inputs['past_ids'] = sampled_ids.to(device)
        model_inputs[self._feat_conf.get("infer_items_key", "sequence_item_ids")] = sampled_ids.to(device)
        return model_inputs    

    def forward(
        self,
        model_inputs,
        positive_ids
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A tuple of (sampled_ids, sampled_negative_embeddings).
        """
        output_shape = positive_ids.size() + (self._num_to_sample,)
        sampled_offsets = torch.randint(
            low=0, high=self._num_items,
            size=output_shape,
            dtype=positive_ids.dtype,
            device=positive_ids.device,
        )
        sampled_ids = self._all_item_ids[sampled_offsets.view(-1)].reshape(output_shape)
        # 负样本直接采样 负样本归一化 负样本只使用item id
        sampled_input = self.process_batches(sampled_ids, model_inputs.copy())
        return sampled_ids, self.normalize_embeddings(self.embedding_module.get_all_item_id_only_embeddings(sampled_input))