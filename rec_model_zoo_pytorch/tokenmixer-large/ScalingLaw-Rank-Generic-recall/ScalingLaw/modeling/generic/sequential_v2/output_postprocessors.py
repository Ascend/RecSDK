import abc

import torch
from typing import Dict, List, Tuple
from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const


class OutputPostprocessorModule(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def forward(
            self,
            output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        pass

@ModelRegistry.register()
class L2NormEmbeddingPostprocessor(OutputPostprocessorModule):
    """
    L2归一化输出后处理模块，用于对模型输出的嵌入进行L2归一化处理。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 64)
        if model_conf.get('use_user_embeddings_for_rerank', False):
            self._embedding_dim *= 2
        self._eps: float = model_cfg[Const.HP].get("eps", 1e-6)

    def forward(
            self,
            output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings = output_embeddings[..., :self._embedding_dim]
        squared_sum = torch.sum(output_embeddings ** 2, dim=-1, keepdim=True)
        return torch.div(output_embeddings, torch.clamp(torch.sqrt(torch.clamp(squared_sum, 0.0) + 1e-10),
                                                        min=self._eps))

@ModelRegistry.register()
class LayerNormEmbeddingPostprocessor(OutputPostprocessorModule):
    """
    层归一化输出后处理模块，用于对模型输出的嵌入进行层归一化处理。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 64)
        self._eps: float = Const.EPS
        self.layer_norm_input = torch.nn.LayerNorm((self._embedding_dim,), eps=self._eps)

    def forward(
            self,
            output_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        output_embeddings = output_embeddings[..., :self._embedding_dim]
        return self.layer_norm_input(output_embeddings)
    