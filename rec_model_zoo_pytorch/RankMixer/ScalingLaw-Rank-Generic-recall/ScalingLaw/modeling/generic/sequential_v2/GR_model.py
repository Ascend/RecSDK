from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional
from regex import F

import torch
from modeling.generic.sequential_v2.attn_mask_modules import AttentionMaskModule
from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.generic.sequential_v2.embedding_modules import EmbeddingModule
from modeling.generic.sequential_v2.input_features_preprocessors import InputFeaturesPreprocessorModule
from modeling.generic.sequential_v2.loss_modules import LossModule
from modeling.generic.sequential_v2.negative_sampler import NegativesSampler
from modeling.generic.sequential_v2.output_postprocessors import OutputPostprocessorModule
from modeling.generic.sequential_v2.prediction_modules import FeedForwardModule
from modeling.generic.sequential_v2.DLRM import DLRModule, RankMixer
from modeling.generic.sequential_v2.transformers import SequentialModule, TransformerCache, TransformerCacheState
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from modeling.generic.sequential_v2.features import SequentialFeatures
from modeling.generic.sequential_v2.utils import get_current_embeddings
from torch.autograd.profiler import record_function
import os
NPU_FLAG = os.environ.get('NPU_FLAG', False)

@ModelRegistry.register(
    req_subs={"EmbeddingModule", "InputFeaturesPreprocessorModule", "SequentialModule", "AttentionMaskModule",
              "FeedForwardModule", "OutputPostprocessorModule", "DLRModule","LossModule"}, opt_subs={"NegativesSampler"})
class GR_model(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self._verbose = model_cfg[Const.HP].get("verbose", True)
        model_conf = common_hp["model_conf"]
        feat_conf = common_hp["feature_conf"]
        max_sequence_len = model_conf["max_sequence_length"]
        max_output_len = model_conf.get("gr_output_length",0)
        infer_timestamps_key = feat_conf.get("infer_timestamps_key", "timestamps")
        infer_items_key = feat_conf.get("infer_items_key", "sequence_item_ids")
        self.use_user_embedding = common_hp["train_conf"].get("use_user_embedding", True)
        self.embedding_module: EmbeddingModule = self.init_sub_model("EmbeddingModule")
        self.input_processor_module: InputFeaturesPreprocessorModule = self.init_sub_model("InputFeaturesPreprocessorModule")
        self.sequence_model: SequentialModule = self.init_sub_model("SequentialModule")
        self._action_as_seperate_token : bool = model_conf.get("action_as_seperate_token", True)
        self.negative_sampler: NegativesSampler = None if "NegativesSampler" not in model_cfg[Const.SUB_MODELS] \
            else self.init_sub_model("NegativesSampler")
        
        if self.negative_sampler is not None:
            self.negative_sampler.load_embedding_module(self.embedding_module)
        self._max_sequence_length: int = max_sequence_len + max_output_len
        self.infer_timestamps_key = infer_timestamps_key
        self.infer_items_key = infer_items_key
        self.concat_user_embeddings = model_conf.get("use_user_embeddings_for_rerank", False)
        self.token_per_item = feat_conf.get("token_per_item", 2)
        self.deep_model: DLRModule = self.init_sub_model("DLRModule")
        self._npu_use_fp16 = common_hp["train_conf"].get("npu_use_fp16", False)
        if NPU_FLAG != "False" and self._npu_use_fp16:
            self.deep_model.to(torch.float16)
        else:
            self.deep_model.to(torch.bfloat16)
        self.attention_mask_module: AttentionMaskModule = self.init_sub_model("AttentionMaskModule")
        self.feed_forward_module: FeedForwardModule = self.init_sub_model("FeedForwardModule")
        self.output_processor_module: OutputPostprocessorModule = self.init_sub_model("OutputPostprocessorModule")
        self.loss_module: LossModule = self.init_sub_model("LossModule")
        self.reset_params()
        
        self.multi_user = common_hp["train_conf"].get("multi_user", False)
        if not self.use_user_embedding:
            self.user_feat_num = 0
        elif self.multi_user:
            user_feat_num = 0
            user_feature_columns = feat_conf["user_feature_columns"]
            for v in user_feature_columns.values():
                if v["enabled"] == True:
                    user_feat_num += 1
            self.user_feat_num = user_feat_num
        else:
            self.user_feat_num = 1
    
    def reset_params(self):
        for name, params in self.named_parameters():
            if ("sequence_model" in name) or ("embedding_module" in name):
                if self._verbose:
                    logging.info("Skipping init for %s", name)
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    logging.info("Initialize %s as xavier normal: %s params", name, params.data.shape[0])
            except Exception:
                if self._verbose:
                    logging.info("Failed to initialize %s: %s params", name, params.data.shape[0])

    def get_embeddings(self, model_inputs):
        past_embeddings = self.embedding_module.get_item_embeddings(model_inputs)
        candidate_embeddings = self.embedding_module.get_candidate_item_embeddings(model_inputs) if not self.training else None
        user_feature_embs = self.embedding_module.get_user_embeddings(model_inputs)
        return past_embeddings, user_feature_embs, candidate_embeddings
    
    def generate_input_seqence(
        self, 
        model_inputs,
        num_rerank: int
    ) -> torch.Tensor:
        """
        综合序列信息，生成user, item1, action1, item2, action2...形式的输入序列。

        :return user_embeddings: 拼接后的输入给模型的token序列，形如user, item1, action1, item2, action2...
        :return past_embeddings: 原始的商品token序列，形如item1, item2, item3, ...
        :return all_timestamps: 时间戳序列
        """
        past_lengths = model_inputs['past_lengths']
        past_ids = model_inputs[self.infer_items_key]
        past_payloads = model_inputs
        past_embeddings, user_feature_embs, candidate_embeddings = self.get_embeddings(model_inputs)
        if NPU_FLAG != "False" and self._npu_use_fp16:
            past_embeddings = past_embeddings.to(torch.float16)
        else:
            past_embeddings = past_embeddings.to(torch.bfloat16)
        
        with record_function("## deep_model ##"):
            deep_results = self.deep_model(
                past_ids=None,
                num_rerank=num_rerank,
                model_inputs=model_inputs,
                user_feature_embs=None,
                item_feature_embs=past_embeddings,
                )
        # 如果是在推理时，将历史序列和候选集序列拼接后返回；如果在训练，则只返回历史序列。
        if (torch.onnx.is_in_onnx_export() or not self.training) and num_rerank > 0:
            past_lengths_after_input_processor, user_embeddings, _ = self.input_processor_module(
                user_feature_embs=user_feature_embs,
                past_lengths=past_lengths,
                past_ids=past_ids,
                past_embeddings=past_embeddings,
                past_payloads=past_payloads,
                num_rerank=num_rerank
            )
            rerank_embeddings = self.input_processor_module.process_rerank_embs(
                rerank_embs=candidate_embeddings,
                past_lengths=past_lengths_after_input_processor
            )
            user_embeddings = torch.concat([user_embeddings, rerank_embeddings], dim=1)
            all_timestamps = torch.concat(
                [past_payloads[self.infer_timestamps_key], past_payloads['candidate_' + self.infer_timestamps_key]], dim=1)
        else:
            past_lengths_after_input_processor, user_embeddings, _ = self.input_processor_module(
                user_feature_embs=user_feature_embs,
                past_lengths=past_lengths,
                past_ids=past_ids,
                past_embeddings=past_embeddings,
                past_payloads=past_payloads,
                num_rerank=num_rerank
            )
            if (all_timestamps := past_payloads.get(self.infer_timestamps_key, None)) is not None:
                all_timestamps = all_timestamps.detach()
        return user_embeddings, past_embeddings, all_timestamps, past_lengths_after_input_processor, deep_results
    
    def generate_user_embeddings(
            self,
            past_lengths: torch.Tensor,
            all_timestamps: torch.Tensor,
            seq_embeddings: torch.Tensor,
            attn_mask: torch.Tensor,
            cache: Optional[List[TransformerCacheState]] = None,
            delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
            return_cache_states: bool = False,
            num_rerank: int = 0,
            x_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        综合序列信息，生成用户 embedding.
        [B, N] -> [B, N, D].
        """
        if x_offsets is None:
            x_offsets = torch.cat((
                torch.full((1,), 0, dtype=past_lengths.dtype).to(past_lengths.device),
                torch.cumsum(past_lengths, dim=0)
            ), dim=0)
            
        item_embeddings, _ = self.sequence_model(
            x=seq_embeddings,
            x_offsets=x_offsets,
            all_timestamps=all_timestamps,
            invalid_attn_mask=attn_mask,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            delta_x_offsets=delta_x_offsets,
            cache=cache,
            return_cache_states=return_cache_states,
        )
        # 如果推理时，只返回候选集的部分的输出的商品token；如果训练时，返回所有商品的输出的token。
        if torch.onnx.is_in_onnx_export() or num_rerank > 0:
            _item_embeddings = item_embeddings[:,-num_rerank-1:-1,:]
        else:
            _item_embeddings = item_embeddings[:,self.user_feat_num:-1:self.token_per_item,:]
        
        if self.concat_user_embeddings:
            user_embeddings = item_embeddings[:,:1,:].repeat(1, _item_embeddings.shape[1], 1)
            item_embeddings = torch.cat([user_embeddings, _item_embeddings], dim=-1)
        else:
            item_embeddings = _item_embeddings
            
        return self.output_processor_module(item_embeddings)
    
    def forward(
            self,
            model_inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        生成式推荐大模型前向传播过程
        
        :param model_inputs: 传入的字典，里面包括商品的特征id序列，用户的特征和其他序列信息。
        """
        # past_lengths: 序列的有效长度（padding前的历史交互数量）
        past_lengths = model_inputs['past_lengths']
        # num_rerank: 候选集商品的数量，训练时因为不会传入candidate_前缀的特征，所以训练时这个值是0
        num_rerank = model_inputs.get('candidate_' + self.infer_items_key, torch.tensor([[]])).shape[1]
        attn_mask = self.attention_mask_module(model_inputs, self._max_sequence_length, num_rerank, num_user_tokens=self.user_feat_num)
        seq_embeddings, past_embeddings, all_timestamps, past_lengths_after_input_processor, deep_results = self.generate_input_seqence(model_inputs=model_inputs, num_rerank=num_rerank)
        encoded_embeddings = self.generate_user_embeddings(past_lengths=past_lengths_after_input_processor,
                                               all_timestamps=all_timestamps,
                                               seq_embeddings=seq_embeddings,
                                               attn_mask=attn_mask,
                                               num_rerank=num_rerank)
        results = self.feed_forward_module(encoded_embeddings)
        results["raw_output"] = encoded_embeddings
        if torch.onnx.is_in_onnx_export() or num_rerank > 0:
            return results
        else:
            past_embeddings = self.embedding_module.get_all_item_id_only_embeddings(model_inputs)
            loss = self.loss_module(past_embeddings=past_embeddings,
                                     encoded_embeddings=encoded_embeddings,
                                     predictions=results,
                                     model_inputs=model_inputs,
                                     negative_sampler=self.negative_sampler)
            aux_loss = torch.sum(deep_results["aux_loss"])
            return loss+aux_loss

    def encode_static_for_compile(self, model_inputs):
        attn_mask = self.attention_mask_module(model_inputs, self._max_sequence_length, num_rerank=0, num_user_tokens=self.user_feat_num)
        seq_embeddings, past_embeddings, all_timestamps, past_lengths_after_input_processor, deep_results = self.generate_input_seqence(model_inputs=model_inputs, num_rerank=0)
        return deep_results
    
    def encode_static(self, model_inputs):
        return self.encode_static_for_compile(model_inputs)

    # 仅供召回使用
    def encode(
            self,
            past_lengths: torch.Tensor,
            past_ids: torch.Tensor,
            past_embeddings: torch.Tensor,
            past_payloads: Dict[str, torch.Tensor],
            model_inputs,
    ) :
        deep_results = self.encode_static(model_inputs)

        actual_sparsity = float(1 - deep_results["deep_sparsity"].detach())
        target_sparsity = 0.1
        alpha = 1.4
        diff = target_sparsity - actual_sparsity
        if diff > 0:
            self.deep_model.balance_loss_coef *= alpha
        elif diff < 0:
            self.deep_model.balance_loss_coef /= alpha
        deep_results["aux_loss"] = self.deep_model.balance_loss_coef * deep_results["deep_loss"]

        return deep_results


@ModelRegistry.register()
class GRModelEp(GR_model):
    
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        model_cfg[Const.SUB_MODELS]["EmbeddingModule"][Const.MODULE_NAME] = "DistributeEmbeddingModuleWithSideInfo"
        super().__init__(model_cfg, common_hp, model_cls_dict)
    
    def get_embeddings(self, model_inputs):
        return self.embedding_module.get_all_embeddings(model_inputs)

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        all_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        attn_mask: torch.Tensor,
        cache: Optional[List[TransformerCacheState]] = None,
        delta_x_offsets: Tuple[torch.Tensor, torch.Tensor] = (torch.tensor([]), torch.tensor([])),
        return_cache_states: bool = False,
        num_rerank: int = 0,
        x_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x_offsets is None:
            x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(past_lengths)

        return super().generate_user_embeddings(past_lengths=past_lengths,
                                                all_timestamps=all_timestamps,
                                                seq_embeddings=seq_embeddings,
                                                attn_mask=attn_mask,
                                                cache=cache,
                                                delta_x_offsets=delta_x_offsets,
                                                return_cache_states=return_cache_states,
                                                num_rerank=num_rerank,
                                                x_offsets=x_offsets)