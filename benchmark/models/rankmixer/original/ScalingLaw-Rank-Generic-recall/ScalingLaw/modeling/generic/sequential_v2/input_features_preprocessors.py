import abc
import math
from typing import Dict, Tuple
import logging
import torch
from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.generic.initialization import truncated_normal
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from utils.common_utils import weird_division
from modeling.generic.initialization import truncated_normal


class InputFeaturesPreprocessorModule(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)

    @abc.abstractmethod
    def process_rerank_embs(
            self,
            rerank_embs: torch.Tensor, 
            past_lengths: torch.Tensor,  # B, 1
    ):
        pass

    @abc.abstractmethod
    def forward(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            user_feature_embs: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pass

@ModelRegistry.register()
class UserItemRatingInputFeaturePreprocessor(InputFeaturesPreprocessorModule):
    """
    用户-物品-评分输入特征预处理模块, 用于处理用户、物品和评分的特征。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        feat_conf = common_hp["feature_conf"]
        max_sequence_length = model_conf.get("max_sequence_length", 512)
        gr_output_length = model_conf.get("gr_output_length", 0)
        max_sequence_len = max_sequence_length + gr_output_length
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)
        self.token_per_item = feat_conf.get("token_per_item", 2)
        self.use_user_embedding = common_hp["train_conf"].get("use_user_embedding", True)
        self.rating_flag = feat_conf.get("rating_flag", 1)
        
        assert self.token_per_item == 1 or self.token_per_item == 2, f"invalid {self.token_per_item}"
        self._pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            max_sequence_len * self.token_per_item + 2, self._embedding_dim,
        )
        
        
        num_ratings = model_conf.get("num_ratings", 5)
        self.num_ratings = num_ratings
        if self.rating_flag:
            self._rating_emb: torch.nn.Embedding = torch.nn.Embedding(
                num_ratings + 1, self._embedding_dim, padding_idx=0
            )
        self._infer_ratings_key = feat_conf.get("infer_ratings_key", "ratings")
        
        self._dropout_rate: float = model_conf.get("linear_dropout_rate", 0.3)
        self._emb_dropout = torch.nn.Dropout(p=self._dropout_rate)
        self.reset_state()

    def debug_str(self) -> str:
        return f"combir_d{self._dropout_rate}"

    def reset_state(self) -> None:
        truncated_normal(
            self._pos_emb.weight.data, mean=0.0, std=math.sqrt(weird_division(1.0, self._embedding_dim)),
        )
        if self.rating_flag:
            truncated_normal(
                self._rating_emb.weight.data, mean=0.0, std=math.sqrt(weird_division(1.0, self._embedding_dim)),
            )
    
    def get_preprocessed_masks(
            self,
            past_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        生成预处理后的掩码。
        
        :param past_ids: 历史ID张量。
        :return: 预处理后的掩码张量, 形状为(B, N * 2)。
        """
        B, N = past_ids.size()
        return (past_ids != 0).unsqueeze(2).expand(-1, -1, self.token_per_item).reshape(B, N * self.token_per_item)
        
    def forward(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            user_feature_embs: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
            num_rerank: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播方法, 用于处理用户、物品和评分的特征。
        
        :param past_lengths: 历史长度张量, 形状为(B,), 其中B是批次大小, 表示每个序列的长度。
        :param past_ids: 历史ID张量, 形状为(B, N), 其中N是序列中的最大项数, 表示每个序列中的物品ID。
        :param user_feature_embs: 用户特征embedding张量, 形状为(B, D), D是用户特征embedding的维度。
        :param past_embeddings: 历史embedding张量, 形状为(B, N, D), 包含序列中每个物品的embedding表示。
        :param past_payloads: 历史信息字典, 包含序列中每个项的额外信息, 如评分和时间戳。
        :return: 新的序列长度, 形状为(B,), 是原始有效序列长度的两倍+1。
                 预处理后的用户特征embedding, 形状为(B, 1+N*2+1, D), 将用户特征、物品特征和评分特征结合起来最后补0。
                 有效掩码张量, 形状为(B, N*2), 用于指示哪些位置是有效的, 即非零ID的位置。
        """
        B, N = past_ids.size()
        D = past_embeddings.size(-1)

        # 提取评分 embedding
        if self.rating_flag:
            rating_emb = self._rating_emb(past_payloads[self._infer_ratings_key])

        # 拼接历史物品 embedding 与评分 embedding, (i1,i2,i3,...), (a1,a2,a3,...)->(i1,a1,i2,a2,i3,a3,...)
        if self.token_per_item == 2: # 2 or 1
            user_embeddings = torch.cat(
                [
                    past_embeddings,
                    rating_emb
                ], dim=2,
            ) * (self._embedding_dim ** 0.5)
            user_embeddings = user_embeddings.view(B, N * self.token_per_item, D)
            user_embeddings = (
                    user_embeddings
                    + self._pos_emb(torch.arange(N * self.token_per_item, device=past_ids.device).unsqueeze(0).repeat(B, 1))
            )
            # seq_len =  1 + past_lengths * self.token_per_item
            if self.use_user_embedding:
                seq_len = 1 + past_lengths * self.token_per_item
            else:
                seq_len = past_lengths * self.token_per_item

        else:
            if self.rating_flag:
                user_embeddings = (past_embeddings + rating_emb) * (self._embedding_dim ** 0.5) + self._pos_emb(torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)) 
            else:
                user_embeddings = (past_embeddings) * (self._embedding_dim ** 0.5) + self._pos_emb(torch.arange(N, device=past_ids.device).unsqueeze(0).repeat(B, 1)) 
            # i1+a1, i2+a2, i3+a3, ...
            # seq_len = 1 + past_lengths
            if self.use_user_embedding:
                seq_len = 1 + past_lengths
            else:
                seq_len = past_lengths
        user_embeddings = self._emb_dropout(user_embeddings)

        # 生成有效掩码并应用
        valid_mask = self.get_preprocessed_masks(
            past_ids
        ).unsqueeze(2).float()
        user_embeddings *= valid_mask

        if self.use_user_embedding:
            user_embeddings = torch.cat((user_feature_embs.unsqueeze(1), user_embeddings), dim=1)

        if num_rerank == 0:
            user_embeddings = torch.nn.functional.pad(user_embeddings, (0, 0, 0, 1, 0, 0), 'constant', 0.0)
        return seq_len, user_embeddings, valid_mask

    def process_rerank_embs(
            self,
            rerank_embs: torch.Tensor,  # B, NUM_CANDIDATE, D
            past_lengths: torch.Tensor,  # B, 1
    ):
        B, N, D = rerank_embs.shape
        position_embs = self._pos_emb(past_lengths - 1).unsqueeze(1).repeat(1, N, 1)  # B x NUM_CANDIDATE x D
        rerank_embs = rerank_embs * (self._embedding_dim ** 0.5) + position_embs
        rerank_embs = torch.nn.functional.pad(rerank_embs, (0, 0, 0, 1, 0, 0), 'constant', 0.0)
        return rerank_embs
    
    def get_num_ratings(self):
        return self.num_ratings
