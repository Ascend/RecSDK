import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Tuple
from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from modeling.generic.utils.almightygosu_utils import index_select

@ModelRegistry.register(req_subs={"RelativeTimeEncoder", "RelativePositionEncoder"})
class RABModule(BaseModel):
    """
    RAB模块

    :param rab_aggregate_method: rab相对时间和相对位置编码聚合的方式，可选'sum'或'concat'或None
    """
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        feat_conf=common_hp["feature_conf"]
        self._max_seq_len: int = model_conf.get("max_sequence_length", 256)
        self._num_layers: int = model_conf.get("num_blocks", 8)
        self.aggregate = model_cfg[Const.HP].get("rab_aggregate_method", 'sum')
        self.token_per_item = feat_conf.get('token_per_item', 2)
        # RAB模块必须在submodel里指定一个时间编码器和一个位置编码器
        if "RelativeTimeEncoder" in model_cfg[Const.SUB_MODELS]:
            self.rel_t_encoder = self.init_sub_model("RelativeTimeEncoder")
        else:
            logging.error("A RelativeTimeEncoder should be assigned in sub_models")
        if "RelativePositionEncoder" in model_cfg[Const.SUB_MODELS]:
            self.rel_p_encoder = self.init_sub_model("RelativePositionEncoder")
        
        

    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = all_timestamps.shape[0]
        rel_ts_bias = self.rel_t_encoder(
            all_timestamps=all_timestamps,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            layer_num=layer_num,
            time_bias=time_bias,
            length=length
        )
        rel_pos_bias = self.rel_p_encoder(
            all_timestamps=all_timestamps,
            past_lengths=past_lengths,
            num_rerank=num_rerank,
            layer_num=layer_num,
            time_bias=time_bias,
            length=length
        )
        
        if torch.onnx.is_in_onnx_export() or num_rerank != 0:
            # 维度 [bs, (length+num_rerank)//2, (length+num_rerank)//2]
            extra = (num_rerank // self.token_per_item) * (self.token_per_item - 1)
            rel_ts_bias = rel_ts_bias.view(bs, length + num_rerank, length + num_rerank)[:, :-extra or None, :-extra or None]  # to improve
            rel_pos_bias = torch.nn.functional.pad(rel_pos_bias, (0, num_rerank // self.token_per_item, 0, num_rerank // self.token_per_item), 'constant',
                                                   0.0)
            # 形如[bs, length+(num_rerank//2), length+(num_rerank//2)]
            rel_pos_bias = rel_pos_bias.repeat(bs, 1, 1)
            rel_pos_bias_lst = []
            for i in range(bs):
                rel_pos_bias_lst.append(torch.where(
                    (torch.arange(rel_pos_bias[i].size(0)).to(past_lengths.device) < (
                            past_lengths[i] * 0.5).int()).unsqueeze(1),
                    rel_pos_bias[i],
                    rel_pos_bias[i, (past_lengths[i] * 0.5).int()].unsqueeze(0)
                ))
            rel_pos_bias = torch.stack(rel_pos_bias_lst)
            max_len = self._max_seq_len + num_rerank // 2
            pos_indices = torch.arange(max_len).repeat(max_len).view(max_len, max_len).to(rel_pos_bias.device)
            identity = (pos_indices.t() == pos_indices).float()
            rel_pos_bias = rel_pos_bias * (1 - identity) + identity * rel_pos_bias[0, 0, 0]
            rel_pos_bias = rel_pos_bias[:, :length + num_rerank // self.token_per_item :length + num_rerank // self.token_per_item]
        if self.aggregate == 'sum':
            return rel_pos_bias + rel_ts_bias, time_bias
        elif self.aggregate == 'concat':
            if num_rerank == 0:
                rel_pos_bias = rel_pos_bias.repeat(bs,1,1)
            return torch.stack([rel_pos_bias, rel_ts_bias], dim=1), time_bias 
        # None, 直接输出
        else:
            return (rel_pos_bias, rel_ts_bias), time_bias



class RelativeTimeEncoder(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
    
    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length=None
    ):
        pass

@ModelRegistry.register()
class BucketRelativeTimeEncoder(RelativeTimeEncoder):
    """
    HSTU论文提出的基于分桶 + 索引的相对时间编码器

    :param num_buckets: 分桶的数量
    :param bucketization_divisor: 分桶时的除数
    :param use_fbgemm: 是否使用fbgemm
    """
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._max_seq_len: int = model_conf.get("max_sequence_length", 256)
        self._num_layers: int = model_conf.get("num_blocks", 8)
        self._num_buckets = model_cfg[Const.HP].get("num_buckets", 48)
        self._bucketization_divisor = model_cfg[Const.HP].get("bucketization_divisor", 0.301)
        self._ts_w = torch.nn.Parameter(
                torch.empty(self._num_buckets + 1).normal_(mean=0, std=0.02),
            ).contiguous()
        self._use_fbgemm = model_cfg[Const.HP].get("use_fbgemm", True)
        # rab_ij = t_{i+1} - t_j or rab_ij = t_i - t_j
        self._use_next_timestamp = model_cfg[Const.HP].get("use_next_timestamp", False)
        
    # 相对时间编码，HSTU论文原始的编码方式，分桶操作
    def bucketization_fn(self, x: torch.Tensor):
        return (torch.log(torch.abs(x.detach()).clamp(min=1)) / self._bucketization_divisor).long()

    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length=None
    ):
        bs = all_timestamps.shape[0]  
        # 只有导出推理模型时会非None
        if length is None:
            length = self._max_seq_len
        # 形如 [bs, n, n]
        N = self._max_seq_len  
        if self._use_next_timestamp:
            all_timestamps = torch.cat([all_timestamps, all_timestamps[:, N - 1 : N]], dim=1)
            timestamp_diffs = all_timestamps[:, 1:].unsqueeze(2) - all_timestamps[:, :-1].unsqueeze(1)
        else:
            timestamp_diffs = all_timestamps.unsqueeze(2) - all_timestamps.unsqueeze(1)
        bucketed_timestamps = torch.clamp(
            self.bucketization_fn(
                (timestamp_diffs).to(torch.float32)),
            min=0,
            max=self._num_buckets,
        ).detach()

        rel_ts_bias = torch.tensor([])
        rel_ts_bias = index_select(x=self._ts_w, index=bucketed_timestamps.view(-1))\
                            .view(bs, length, length)
        return rel_ts_bias

@ModelRegistry.register()
class PowRelativeTimeEncoder(RelativeTimeEncoder):
    """
    fuxi-beta论文提出的相对时间编码，基于指数函数拟合
    """
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        self._a = torch.nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(-0.2, 0.2))
        self._d = torch.nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.2, 1.5))
        self._e = torch.nn.Parameter(torch.empty(1, dtype=torch.float32).uniform_(0.4, 0.8))

    def f(self, x) :
        x = torch.relu(x) + 1
        return self._a / (1 + self._d * torch.pow(x, self._e))
    
    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length=None
    ):
        # 只有导出推理模型时会非None
        if length is None:
            length = self._max_seq_len  
        ext_timestamps = (all_timestamps.unsqueeze(2) - all_timestamps.unsqueeze(1)).to(torch.float32)
        rel_ts_bias = self.f(ext_timestamps)
        return rel_ts_bias
    
class RelativePositionEncoder(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
    
    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length=None
    ):
        pass

@ModelRegistry.register()
class DefaultRelativePositionEncoder(RelativePositionEncoder):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._max_seq_len: int = model_conf.get("max_sequence_length", 256)
        self._num_layers: int = model_conf.get("num_blocks", 8)
        self._pos_w = torch.nn.Parameter(
            torch.empty(2 * (self._max_seq_len + 1) - 1).normal_(mean=0, std=0.02),
        )
    def forward(
            self,
            all_timestamps: torch.Tensor,
            past_lengths: torch.Tensor,
            num_rerank: int,
            layer_num: int,
            time_bias: torch.Tensor = torch.tensor([]),
            length=None
    ):
        # 相对位置编码
        # 选择对应层的位置权重，length * (2*length-1 + length) = 3 * length**2 - length
        t = F.pad(self._pos_w[:2 * self._max_seq_len - 1], [0, self._max_seq_len]).repeat(self._max_seq_len)
        # 去掉 t 最后 length 个填充的零, 1 x length x 3*length-2)
        # 例如
        # ：[1, 2, 3, 4, 5, 0, 0]
        # ：[0, 1, 2, 3, 4, 5, 0]
        # ：[0, 0, 1, 2, 3, 4, 5]
        t = t[..., :-self._max_seq_len].reshape(1, self._max_seq_len, 3 * self._max_seq_len - 2)
        # 计算中心位置的索引
        r = (2 * self._max_seq_len - 1) // 2
        # 从 t 中提取中心位置 r 两侧的位置偏置, 形如[1, length, length]
        # 获得pos_w的循环排列
        # ：[3, 4, 5]
        # ：[2, 3, 4]
        # ：[1, 2, 3]
        rel_pos_bias = t[:, :, r:-r] # 1, N, N
        return rel_pos_bias
    