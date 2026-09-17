from typing import Dict
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from modeling.generic.sequential_v2.features import SequentialFeatures

@ModelRegistry.register(multi_sel_multi_subs=[{"CausalAttentionMask", "TimeAttentionMask"}])
class AttentionMaskModule(BaseModel):
    """
        "AttentionMaskModule": {
            "type": ["CausalAttentionMask", "TimeAttentionMask"],
            "cfg": {}
        }

        :param model_conf:
        :param model_factory:
        """
    
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self.attention_mask_modules = nn.ModuleList([
            self.init_sub_model(sub_key)
            for sub_key in model_cfg[Const.SUB_MODELS].keys()
 
        ])
        self.use_user_embedding = common_hp["train_conf"].get("use_user_embedding", True)
    
    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank, num_user_tokens):
        init_mask = None
        if not self.use_user_embedding:
            num_user_tokens = 0
        for attn_module in self.attention_mask_modules: 
            mask = attn_module(model_inputs=model_inputs, max_seq_len=max_seq_len, num_rerank=num_rerank, num_user_tokens=num_user_tokens)
            if init_mask is None:
                    init_mask = mask
            else:
                init_mask = init_mask * mask
        attn_mask = init_mask.detach().clone()
        return attn_mask
    
@ModelRegistry.register()
class CausalAttentionMask(BaseModel):
    
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        因果注意力掩码生成
        训练时：生成形状为(bs, 2 * max_seq_len + 2, 2 * max_seq_len + 2)的下三角矩阵
        推理时：生成形状为(bs, 2 * max_seq_len + 2 + num_rerank, 2 * max_seq_len + 2 + num_rerank)的掩码矩阵，
        其中前2 * max_seq_len + 2行/列与训练时的掩码矩阵相同，但候选集部分token互相不可见
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp["feature_conf"]
        model_conf = common_hp["model_conf"]
        self.mask_candidates = model_cfg[Const.HP].get("mask_candidates", True)
        self._infer_timestamps_key = feat_conf.get("infer_timestamps_key", "timestamps")
        self._infer_items_key = feat_conf.get('infer_items_key', 'item_id')
        self.token_per_item = feat_conf.get("token_per_item", 2)

    def init_mask_for_export(self, seq_len, num_rerank, device, max_len):
        #max_len = seq_len * 2 + 2 + num_rerank
        _pos_indices = torch.arange(max_len).repeat(max_len).view(max_len, max_len).to(device)
        _base_mask = (_pos_indices.t() > _pos_indices).float()
        _identity = (_pos_indices.t() == _pos_indices).float()
        return _pos_indices, _identity, _base_mask

    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank, num_user_tokens):
        device = model_inputs[self._infer_timestamps_key].device
        max_len = max_seq_len * self.token_per_item + num_rerank + 1 + num_user_tokens
            
        if not self.mask_candidates:
            indices = torch.arange(max_len, device=device)
            t = indices.expand(max_len, max_len)
            attn_mask = (t.t() >= indices).unsqueeze(0)
            return attn_mask
        else:
            if num_rerank == 0:
                indices = torch.arange(max_len, device=device)
                t = indices.expand(max_len, max_len)
                attn_mask = (t.t() >= indices).unsqueeze(0) 
            elif torch.onnx.is_in_onnx_export() or num_rerank > 0:
                past_lengths = model_inputs['past_lengths']
                past_lengths = num_user_tokens + past_lengths * self.token_per_item
                _past_lengths = past_lengths.unsqueeze(-1).unsqueeze(-1)
                _pos_indices, _identity, _base_mask = self.init_mask_for_export(seq_len=max_seq_len, num_rerank=num_rerank, device=device, max_len=max_len)
                seq_mask = (_pos_indices < _past_lengths).int()
                attn_mask = (seq_mask * _base_mask + _identity)
            return attn_mask


@ModelRegistry.register()
class TimeAttentionMask(BaseModel):
    
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        时间注意力掩码生成
        生成形状为(bs, 2 * max_seq_len + 2 + num_rerank, 2 * max_seq_len + 2 + num_rerank)的掩码矩阵，
        其中每个token的可见范围由timestamp_mask_threshold确定。
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        feat_conf = common_hp["feature_conf"]
        model_conf = common_hp["model_conf"]
        self._infer_timestamps_key = feat_conf.get("infer_timestamps_key", "timestamps")
        self._time_threshold = model_cfg[Const.HP].get('timestamp_mask_threshold', 86400)
        self._infer_items_key = feat_conf.get('infer_items_key', 'item_id')
        self.token_per_item = feat_conf.get("token_per_item", 2)
          
    def forward(self, model_inputs: Dict[str, torch.Tensor], max_seq_len, num_rerank, num_user_tokens):
        device = model_inputs[self._infer_timestamps_key].device
        bs = model_inputs.get(self._infer_items_key).shape[0]
        all_timestamps = model_inputs[self._infer_timestamps_key].detach()
        
        time_threshold_mask = ((all_timestamps.unsqueeze(2) - all_timestamps.unsqueeze(1)) <= self._time_threshold).to(torch.float32)
        
        _pos_indices = torch.arange(max_seq_len).repeat(max_seq_len).view(max_seq_len, max_seq_len).to(device)
        _identity = (_pos_indices.t() == _pos_indices).float()
        time_threshold_mask = time_threshold_mask - _identity
        attn_mask = (
            1.0
            - time_threshold_mask
                .unsqueeze(1).unsqueeze(-1)
                .repeat(1, 1, 1, self.token_per_item, self.token_per_item)
                .reshape(bs, max_seq_len * self.token_per_item, max_seq_len * self.token_per_item)
        )

        attn_mask = F.pad(attn_mask, (num_user_tokens, 1+num_rerank, num_user_tokens, 1+num_rerank), 'constant', 1.0)
        
        return attn_mask