from numpy import add
import torch
ENABLE_NPU=False
try:
    import torch_npu
    ENABLE_NPU=True
except:
    import torch
    import grouped_gemm

import torch.nn.functional as F
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.generic.sequential_v2.transformers import RMSNormNPU
from modeling.generic.initialization import truncated_normal
from modeling.model_registry import ModelRegistry
from modeling.generic.utils.constants import Const
from torch.autograd.profiler import record_function

import logging
torch._dynamo.disallow_in_graph(F.one_hot)

@ModelRegistry.register(req_subs={"RankMixer"})
class DLRModule(BaseModel):

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)
        self.dlrm_modules = nn.ModuleList([
            self.init_sub_model(sub_key)
            for sub_key in model_cfg[Const.SUB_MODELS].keys()
        ])
        self.balance_loss_coef = 1e-8

    def forward(self, 
                past_ids,
                num_rerank,
                model_inputs,
                user_feature_embs,
                item_feature_embs):
        init_dlrm_result = None
        init_dlrm_output = 0.0
        init_dlrm_loss = 0.0
        init_dlrm_sparsity = 0.0
        for dlrm_module in self.dlrm_modules:
            dlrm_result = dlrm_module(past_ids=past_ids, num_rerank=num_rerank, model_inputs=model_inputs,
                                     user_feature_embs=user_feature_embs, item_feature_embs=item_feature_embs)
            dlrm_output, dlrm_loss, dlrm_sparsity = dlrm_result["deep_outputs"], dlrm_result["deep_loss"], dlrm_result["deep_sparsity"]
            if init_dlrm_result is None:
                init_dlrm_result = dlrm_result
                init_dlrm_output = dlrm_output
                init_dlrm_loss = dlrm_loss
                init_dlrm_sparsity = dlrm_sparsity
            else:
                # Need to make sure all outputs have same dim.
                init_dlrm_output = init_dlrm_output + dlrm_output
                init_dlrm_loss = init_dlrm_loss + dlrm_loss
                init_dlrm_sparsity = init_dlrm_sparsity + dlrm_sparsity
        return {"deep_outputs": init_dlrm_output, "deep_loss": init_dlrm_loss, "deep_sparsity": init_dlrm_sparsity, "aux_loss": None}

@ModelRegistry.register()
class RankMixingInput(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        dim_per_token = model_cfg[Const.HP].get("dim_per_token")
        x_dim = model_cfg[Const.HP].get("x_dim")
        k = model_cfg[Const.HP].get("k")
        k_multiplier = model_cfg[Const.HP].get("k_multiplier", 1)
        assert (x_dim % dim_per_token) == 0
        T = x_dim // dim_per_token
        num_heads = T
        inner_dim = int(num_heads * k * k_multiplier)
        assert (inner_dim % num_heads) == 0
        self.x_dim = x_dim
        self.num_heads = num_heads
        self.dim_per_token = dim_per_token
        self.inner_dim = inner_dim
        self.proj = torch.nn.Linear(dim_per_token, inner_dim, bias=False)
    def forward(self, x):
        # 输入x是所有用户、商品、序列特征拼接而成的特征向量，形状为(B, \sum_{D_e}), D_e是不同特征的长度
        B, D = x.size()
        assert self.x_dim == D
        x = x[:, :self.num_heads * self.dim_per_token]
        # 形状（B, T, inner_dim）
        proj_x = self.proj(x.view(B, self.num_heads, self.dim_per_token))
        return proj_x

@ModelRegistry.register()        
class TokenMixing(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        """
        RankMixer中的TokenMixing模块,
        对应原论文公式2,3,4,5
        dim_per_token：公式2里的d，即重新划分的token的维度
        x_dim: 所有特征拼接起来的维度总和
        为避免信息损失，要求x_dim可以被dim_per_token整除。
        """
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        dim_per_token = model_cfg[Const.HP].get("dim_per_token")
        x_dim = model_cfg[Const.HP].get("x_dim")
        k = model_cfg[Const.HP].get("k")
        k_multiplier = model_cfg[Const.HP].get("k_multiplier")
        assert (x_dim % dim_per_token) == 0
        T = x_dim // dim_per_token
        num_heads = T
        inner_dim = int(num_heads * k * k_multiplier)
        assert (inner_dim % num_heads) == 0
        self.num_heads = num_heads
        self.ln = torch.nn.LayerNorm((inner_dim,), eps=1e-7)
        
    def forward(self, x):
        # 对每个头（或整体）学习一个 token mixing matrix M ∈ ℝ^{T×T}
        # X' = M * X，直接把 token 信息线性组合到其他 token
        B, D = x.size(0), x.size(2)
        tm_x = x.transpose(1, 2).contiguous().view(B, self.num_heads, D)
        # 形状（B, num_heads, inner_dim）
        return self.ln(x + tm_x)

@ModelRegistry.register()    
class PerTokenFFN(BaseModel):
    """
    Per-token position-wise MLP with configurable depth L.
    Each token position t has its own stack of Linear layers.

    For L layers:
      layer 1: D -> kD
      layer 2..L-1: kD -> kD
      layer L: kD -> D
    Activations: GELU after layers 1..L-1 (no activation on last).
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        D = model_cfg[Const.HP].get("D")
        T = model_cfg[Const.HP].get("T")
        k = model_cfg[Const.HP].get("k")
        num_layers = model_cfg[Const.HP].get("num_layers")
        bias = model_cfg[Const.HP].get("bias")
        dropout_p = model_cfg[Const.HP].get("dropout_p")
        sparseMOE = model_cfg[Const.HP].get("sparseMOE")
        num_experts_per_token = model_cfg[Const.HP].get("num_experts_per_token")

        assert num_layers >= 2, "num_layers must be >= 2"

        self.D = D
        self.T = T
        self.num_experts = num_experts_per_token
        self.kD = int(round(k * D))
        self.num_layers = num_layers
        self.sparseMOE = sparseMOE
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # Build per-position weights for each layer
        in_dims = [D] + [self.kD] * (num_layers - 1)
        out_dims = [self.kD] * (num_layers - 1) + [D]
        # out_dims length is num_layers; first num_layers-1 are kD, last is D

        if self.sparseMOE:
            self.W = nn.ModuleList([
                nn.ParameterList([
                    nn.Parameter(torch.empty(din, dout))
                    for din, dout in zip(in_dims, out_dims)
                ])
                for _ in range(self.num_experts)    # E experts
            ])

            if bias:
                self.b = nn.ModuleList([
                    nn.ParameterList([
                        nn.Parameter(torch.empty(dout))
                        for dout in out_dims
                    ])
                    for _ in range(self.num_experts)
                ])
            else:
                self.b = None

            # gating per expert
            self.gate_W = nn.ParameterList([
                nn.Parameter(torch.empty(out_dims[-1], 1))
                for _ in range(self.num_experts)
            ])
            self.gate_b = nn.ParameterList([
                nn.Parameter(torch.empty(1))
                for _ in range(self.num_experts)
            ])
            self.reset_parameters_sparse()
        else:
            self.W = nn.ParameterList([
                nn.Parameter(torch.empty(T, din, dout))
                for din, dout in zip(in_dims, out_dims)
            ])
            if bias:
                self.b = nn.ParameterList([
                    nn.Parameter(torch.empty(T, dout))
                    for dout in out_dims
                ])
            else:
                self.b = None
            self.gate_W = nn.Parameter(torch.empty(T, out_dims[-1], 1))
            self.gate_b = nn.Parameter(torch.empty(T, 1))
            self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_layers):
            nn.init.xavier_normal_(self.W[i])
            if self.b is not None:
                nn.init.zeros_(self.b[i])
        if hasattr(self, "gate_W"):
            nn.init.xavier_normal_(self.gate_W)
        if hasattr(self, "gate_b"):
            nn.init.zeros_(self.gate_b)
    
    def reset_parameters_sparse(self):
        for e in range(self.num_experts):
            for i in range(self.num_layers):
                nn.init.xavier_normal_(self.W[e][i])
                if self.b is not None:
                    nn.init.zeros_(self.b[e][i])
        if hasattr(self, "gate_W"):
            for e in range(self.num_experts):
                nn.init.xavier_normal_(self.gate_W[e])
        if hasattr(self, "gate_b"):
            for e in range(self.num_experts):
                nn.init.zeros_(self.gate_b[e])


    def forward(self, s: torch.Tensor, gate_t: Optional[torch.Tensor] = None, expert_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            s: (b, D)  ->  v: (b, D)
            """
            if self.sparseMOE:
                bs, D = s.shape
                device = s.device
                s = s.to(torch.float16)
                x = torch.zeros(bs, D, device=device, dtype=s.dtype)
                experts_unique = torch.unique(expert_indices)
                with record_function("## PFFN ##"):
                    for exp_id in experts_unique.tolist():
                        mask = (expert_indices == exp_id).any(dim=-1)  # (bs,)
                        if not mask.any():
                            continue
                        idx = mask.nonzero(as_tuple=False).squeeze(1)  # 激活样本索引
                        x_t = torch.index_select(s, 0, idx).to(torch.float16)            # (k, D)
                        for i in range(self.num_layers):
                            if self.b is not None:
                                x_t = torch.addmm(self.b[exp_id][i], x_t, self.W[exp_id][i])
                            else:
                                x_t = torch.matmul(x_t, self.W[exp_id][i])  # (k, dout)
                            if i < self.num_layers - 1:
                                x_t = F.gelu(x_t)
                                x_t = self.dropout(x_t)
                        g = torch.addmm(self.gate_b[exp_id], x_t, self.gate_W[exp_id])
                        g = torch.sigmoid(g)
                        x_t = x_t * g
                        x_t = self.dropout(x_t)
                        
                        r = gate_t[idx, exp_id].unsqueeze(-1)
                        x_t = x_t * r
                        x.index_add_(0, idx, x_t)
            else:
                bs, T, D = s.shape
                assert T == self.T and D == self.D, f"expected (bs,{self.T},{self.D}), got {tuple(s.shape)}"
                x = s
                # Apply layers 0..L-2 with GELU, final layer without activation
                for i in range(self.num_layers):
                    x = torch.einsum("bti,tid->btd", x, self.W[i]) 
                    if self.b is not None:
                        x = x + self.b[i]
                    if i < self.num_layers - 1:
                        x = F.gelu(x)
                        x = self.dropout(x)
                gates = torch.einsum("btd,tde->bte", x, self.gate_W) + self.gate_b  # (bs, T, 1)
                gates = torch.sigmoid(gates)
                x = x * gates.expand(-1, -1, D)
                # optional dropout on output (comment out if you want *exact* equations)
                x = self.dropout(x)
            return x

@ModelRegistry.register(req_subs={"PerTokenFFN"})    
class SparseMoE(BaseModel):
    """
    ReLU-Routed Sparse MoE with per-token experts.

    Given router h(·) and experts e_j(·):
        G_{i,j} = ReLU(h(s_i))
        v_i     = sum_{j=1..N_e} G_{i,j} * e_j(s_i)

    Args:
        num_experts_per_token (int): N_e, number of experts per token.
        inner_dim (int): D, hidden size per token.
        num_tokens (int): T, number of token positions.
        k, bias, dropout_p: forwarded to PerTokenFFN.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        
        num_experts_per_token = model_cfg[Const.HP].get("num_experts_per_token")
        inner_dim = model_cfg[Const.HP].get("inner_dim")
        num_tokens = model_cfg[Const.HP].get("num_tokens")
        k = model_cfg[Const.HP].get("k")
        num_layers_per_expert = model_cfg[Const.HP].get("num_layers_per_expert")
        bias = model_cfg[Const.HP].get("bias")
        dropout_p = model_cfg[Const.HP].get("dropout_p")
        target_activation_ratio = model_cfg[Const.HP].get("target_activation_ratio")
        sparseMOE = model_cfg[Const.HP].get("sparseMOE")
        top_k_number = model_cfg[Const.HP].get("top_k_number")

        self.D = inner_dim
        self.T = num_tokens
        self.Ne = num_experts_per_token
        self.relu_threshold = 1e-4
        self.target_activation_ratio = target_activation_ratio
        self.sparseMOE = sparseMOE
        self.top_k = top_k_number

        # Router h(·): shared across positions, maps R^D -> R^{N_e}
        self.router = nn.Linear(self.D, self.Ne, bias=False)

        # Experts: N_e copies of PerTokenFFN (each is position-specific over T)
        self.model_cfg[Const.SUB_MODELS]["PerTokenFFN"][Const.HP] = {"D": inner_dim,
                                                                    "T": num_tokens,
                                                                    "k": k,
                                                                    "num_layers": num_layers_per_expert,
                                                                    "bias": bias,
                                                                    "dropout_p": dropout_p,
                                                                    "sparseMOE": sparseMOE,
                                                                    "num_experts_per_token": num_experts_per_token}
        self.token_experts = nn.ModuleList(
            [self.init_sub_model("PerTokenFFN") for _ in range(self.T)]
        )
        self.ln = torch.nn.LayerNorm((inner_dim,), eps=1e-7)
        self.reset_parameters()    

    def reset_parameters(self):
        nn.init.xavier_normal_(self.router.weight)

    def apply_activation_threshold(self, gates: torch.Tensor, target_activation_ratio: float) -> torch.Tensor:
        """
        动态计算门控阈值，过滤门控值以达到目标激活比例
        
        Args:
            gates: 原始门控值，形状为 (bs, T, Ne)，已通过 ReLU 激活（非负）
            target_activation_ratio: 目标激活比例(1.0=全激活,0.5=1/2激活,0.25=1/4激活)
        
        Returns:
            filtered_gates: 过滤后的门控值，形状与输入相同，低于阈值的元素被置为0
        """
        assert 0 < target_activation_ratio <= 1.0, "目标激活比例必须在 (0, 1] 范围内"
        bs, T, Ne = gates.shape
        total_pairs = bs * T * Ne
        target_activated = int(total_pairs * target_activation_ratio)
        if target_activation_ratio == 1.0:
            return gates
        gates_flat = gates.view(-1) 
        sorted_gates, _ = torch.sort(gates_flat, descending=True) 
        if target_activated <= 0:
            return torch.zeros_like(gates)
        threshold_idx = min(target_activated - 1, len(sorted_gates) - 1)
        threshold = sorted_gates[threshold_idx]

        filtered_gates = torch.where(gates >= threshold, gates, torch.tensor(0.0, device=gates.device))
        return filtered_gates
    
    def apply_activation_threshold_tokenwise(
        self, gates: torch.Tensor, target_activation_ratio: float
    ) -> torch.Tensor:
        """
        对每个 token 独立选择稀疏专家(top-k)

        gates: (bs, T, Ne)，非负
        target_activation_ratio: 激活比例，例如 0.25 表示激活 25% 专家。
        """
        assert 0 < target_activation_ratio <= 1.0

        bs, T, Ne = gates.shape

        k = max(1, int(Ne * target_activation_ratio))
        if k >= Ne:
            return gates
        # reshape 成 (bs*T, Ne) 方便处理
        gates2d = gates.view(-1, Ne)  # shape: (N, Ne), N=bs*T
        # 对每个 token 取 top-k
        topk_vals, topk_idx = torch.topk(gates2d, k, dim=1)
        # 构造 mask
        mask = torch.zeros_like(gates2d, dtype=torch.bool)
        mask.scatter_(1, topk_idx, True)
        # 掩码：top-k 保留，其他置 0
        filtered = torch.where(mask, gates2d, gates2d.new_zeros(()))
        return filtered.view(bs, T, Ne)    

    def forward(self, s: torch.Tensor):
        """
        Args:
            s: (bs, T, D)

        Returns:
            v: (bs, T, D)  -- routed mixture of experts output
        """
        assert s.dim() == 3 and s.size(1) == self.T and s.size(2) == self.D, \
            f"expected (bs,{self.T},{self.D}), got {tuple(s.shape)}"
        
        if self.sparseMOE:
            bs, T, D = s.shape
            output_dim = self.D
            # Router logits -> ReLU gates (no softmax, no top-k)
            # shape: (bs, T, N_e)
            with record_function("## gates ##"):
                router_logits = self.router(s) ## [bs, T, E]
                # Top-k token-wise gating
                topk_scores, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1) # shape: [bs, T, top_k]
                topk_gates = F.softmax(topk_scores, dim=-1)
                # build dispatch mask: [bs, T, E]
                gates = torch.zeros_like(router_logits)
                gates.scatter_(dim=-1, index=topk_indices, src=topk_gates)
                float_mask = (gates > 0).float().detach()
                reg_loss = gates.sum(-1).sum(-1)
                # 原生接近于1/2激活的稀疏度
                sparsity = float_mask.sum() / (T * self.Ne)

            with record_function("## Sparse Moe ##"):
                v = torch.zeros(bs, T, output_dim, device=s.device)
                for t in range(T):
                    s_t = s[:, t, :]              # [bs, D]
                    gate_t = gates[:, t, :]       # [bs, E]
                    experts_t = topk_indices[:, t, :]  # [bs, top_k]
                    # call PerTokenFFN for this token
                    out_t = self.token_experts[t](s_t, gate_t, experts_t)  # [bs, D]
                    v[:, t, :] = out_t
        else:
            T = s.size(1)
            # device = s.device
            # Router logits -> ReLU gates (no softmax, no top-k)
            # shape: (bs, T, N_e)
            # gates = F.relu(self.router(s))

            router_logits = self.router(s) ## [bs, T, E]
            # Top-k token-wise gating
            topk_scores, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1) # shape: [bs, T, top_k]
            topk_gates = F.softmax(topk_scores, dim=-1)
            # build dispatch mask: [bs, T, E]
            gates = torch.zeros_like(router_logits)
            gates.scatter_(dim=-1, index=topk_indices, src=topk_gates)
            
            float_mask = (gates > 0).float().detach()
            reg_loss = gates.sum(-1).sum(-1)
            sparsity = float_mask.sum() / (T * self.Ne)
            # optional tiny clamp to avoid pathological NaNs in some setups
            # gates = torch.clamp_min(gates, 0.0)
            expert_outputs = torch.stack([exp(s) for exp in self.token_experts], dim=2)
            # Weighted sum over experts: v[b,t,d] = sum_j gates[b,t,j] * out[b,t,j,d]
            v = torch.einsum("btjd,btj->btd", expert_outputs, gates)
        v = v.to(torch.float16)
        s = s.to(torch.float16)
        return self.ln(s + v), (reg_loss, sparsity)

@ModelRegistry.register()    
class SparseMoE_Gemm(BaseModel):
    """
    ReLU-Routed Sparse MoE with per-token experts.

    Given router h(·) and experts e_j(·):
        G_{i,j} = ReLU(h(s_i))
        v_i     = sum_{j=1..N_e} G_{i,j} * e_j(s_i)

    Args:
        num_experts_per_token (int): N_e, number of experts per token.
        inner_dim (int): D, hidden size per token.
        num_tokens (int): T, number of token positions.
        k, bias, dropout_p: forwarded to PerTokenFFN.
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        
        num_experts_per_token = model_cfg[Const.HP].get("num_experts_per_token")
        inner_dim = model_cfg[Const.HP].get("inner_dim")
        num_tokens = model_cfg[Const.HP].get("num_tokens")
        k = model_cfg[Const.HP].get("k")
        num_layers_per_expert = model_cfg[Const.HP].get("num_layers_per_expert")
        bias = model_cfg[Const.HP].get("bias")
        dropout_p = model_cfg[Const.HP].get("dropout_p")
        target_activation_ratio = model_cfg[Const.HP].get("target_activation_ratio")
        sparseMOE = model_cfg[Const.HP].get("sparseMOE")
        top_k_number = model_cfg[Const.HP].get("top_k_number")
        npu_use_fp16 = common_hp["train_conf"].get("npu_use_fp16", False)

        self.D = inner_dim
        self.T = num_tokens
        self.kD = int(round(k * self.D))
        self.Ne = num_experts_per_token
        self.relu_threshold = 1e-4
        self.target_activation_ratio = target_activation_ratio
        self.sparseMOE = sparseMOE
        self.top_k = top_k_number
        D = self.D
        self.num_layers = num_layers_per_expert
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self._npu_use_fp16 = npu_use_fp16
        
        # Router h(·): shared across positions, maps R^D -> R^{N_e}
        self.router = nn.Linear(self.D, self.Ne, bias=False)
        in_dims = [D] + [self.kD] * (self.num_layers - 1)
        out_dims = [self.kD] * (self.num_layers - 1) + [D]
        
        # -------------------------------------------------------------
        # (T × E × L) FFN weights
        # -------------------------------------------------------------
        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(num_tokens * num_experts_per_token, din, dout)) 
            for din, dout in zip(in_dims, out_dims)
        ])
        if bias:
            self.b = nn.ParameterList([
                nn.Parameter(torch.empty(num_tokens * num_experts_per_token, dout)) 
                for dout in out_dims
            ])
        else:
            self.b = None
        # -------------------------------------------------------------
        # (T × E) gating
        #   gate_W[t*E+e] : (d_L, 1)
        #   gate_b[t*E+e] : (1,)
        # -------------------------------------------------------------
        self.gate_W = nn.Parameter(torch.empty(num_tokens * num_experts_per_token, out_dims[-1], 1)) 
        self.gate_b = nn.Parameter(torch.empty(num_tokens * num_experts_per_token, 1))

        self.ln = torch.nn.LayerNorm((inner_dim,), eps=1e-7)
        self.reset_parameters()
    
    def reset_parameters(self):
        for i in range(self.num_layers):
            nn.init.xavier_normal_(self.W[i])
            if self.b is not None:
                nn.init.zeros_(self.b[i])
        if hasattr(self, "gate_W"):
            nn.init.xavier_normal_(self.gate_W)
        if hasattr(self, "gate_b"):
            nn.init.zeros_(self.gate_b)
    def cutlas_gemm(self, a, b, bias, group_list):
        # [A] @ [B] + [bias]        
        matmul_out = grouped_gemm.ops.gmm(a, b, group_list)
        return matmul_out
    
    def forward(self, s: torch.Tensor):
        """
        Args:
            s: (bs, T, D)

        Returns:
            v: (bs, T, D)  -- routed mixture of experts output
        """
        assert s.dim() == 3 and s.size(1) == self.T and s.size(2) == self.D, \
            f"expected (bs,{self.T},{self.D}), got {tuple(s.shape)}"
        K = self.top_k
        bs, T, D = s.shape
        output_dim = self.D
        x = s.clone()
        # Router logits -> ReLU gates (no softmax, no top-k)
        # shape: (bs, T, N_e)
        with record_function("## gates ##"):
            router_logits = self.router(s) ## [bs, T, E]
            # Top-k token-wise gating
            router_logits = F.softmax(router_logits, dim=-1) # shape: [bs, T, top_k]
            topk_gates, topk_indices = torch.topk(router_logits, k=self.top_k, dim=-1) # shape: [bs, T, top_k]
            bs, T, top_k = topk_indices.shape
            offset = torch.arange(T, device=topk_indices.device).unsqueeze(0).unsqueeze(-1) * self.Ne  # shape: (1, T, 1)
            topk_indices_offset = topk_indices + offset  # shape: (bs, T, top_k)
            expert_mask = F.one_hot(topk_indices_offset, num_classes=T*self.Ne).sum(dim=2)  # shape: (bs, T, T*Ne)
            expert_mask = expert_mask.sum(dim=1) # (b*s, expert_number*T)
            expert_mask = expert_mask.permute(1, 0) # (expert_number*T, b*s)
            expert_mask = expert_mask.sum(dim=1) # (expert_number*T,)
            cuda_group_list = None
            if not ENABLE_NPU:
                cuda_group_list = expert_mask
                cuda_group_list = cuda_group_list.cpu()
            group_list = expert_mask.cumsum(dim=0)

            router_weights = (topk_gates / topk_gates.sum(dim=-1, keepdim=True)).reshape(-1, 1)
            # build dispatch mask: [bs, T, E]
            gates = torch.zeros_like(router_logits)
            gates.scatter_(dim=-1, index=topk_indices, src=topk_gates)
            float_mask = (gates > 0).float().detach()
            reg_loss = gates.sum(-1).sum(-1)
            # 原生接近于1/2激活的稀疏度
            sparsity = float_mask.sum() / (T * self.Ne)

        with record_function("## Group Moe ##"):
            tokens = s.reshape(bs*T, D)
            topk_indices_flat = topk_indices.reshape(bs*T, K)  # [bs*T,k]
            if ENABLE_NPU:
                expend_x, idx = torch_npu.npu_moe_token_permute(
                    tokens=tokens,
                    indices=topk_indices_flat,
                    num_out_tokens=bs*T*K
                )
            else:
                expend_x, idx = grouped_gemm.ops.permute(tokens, topk_indices_flat, bs*T*K)
            h = expend_x #[bs*T*K, D]
            # ---  FFN layers using original W/b ---
            for l in range(self.num_layers):
                with record_function("## PFFN layers ##"):
                    # gather W/b for current group_list
                    w = [self.W[l]]
                    b = [self.b[l]] if self.b else None
                    if ENABLE_NPU:
                        h = torch_npu.npu_grouped_matmul(
                            x=[h],
                            weight=w,
                            group_list=group_list,
                            split_item=2,
                            group_list_type=0,
                            group_type=0,
                            output_dtype=h.dtype
                        )[0]
                    else:
                        h = self.cutlas_gemm(h, self.W[l], self.b[l] if self.b else None, cuda_group_list)
                    if l < self.num_layers-1:
                        h = F.relu(h)
                        h = self.dropout(h)

            with record_function("## sigmoid gate ##"):
                # ---  Token-wise sigmoid gate ---
                gate_W = [self.gate_W]
                gate_b = [self.gate_b]
                if ENABLE_NPU:
                    h_gate = torch_npu.npu_grouped_matmul(
                        x=[expend_x],
                        weight=gate_W,
                        group_list=group_list,
                        split_item=2,
                        group_list_type=0,
                        group_type=0,
                        output_dtype=h.dtype
                    )[0]
                else:
                    h_gate = self.cutlas_gemm(expend_x, self.gate_W, self.gate_b, cuda_group_list)
                gate_scores = torch.sigmoid(h_gate)
            h = h * gate_scores
            if ENABLE_NPU:
                v = torch_npu.npu_moe_token_unpermute(h, idx, router_weights)
            else:
                v = grouped_gemm.ops.unpermute(h, idx, router_weights)
            v = v.view(bs,T,K,D).sum(dim=2)
        if ENABLE_NPU and self._npu_use_fp16:
            v = v.to(torch.float16)
            s = s.to(torch.float16)
        else:
            v = v.to(torch.bfloat16)
            s = s.to(torch.bfloat16)
        return self.ln(s + v), (reg_loss, sparsity)


@ModelRegistry.register(req_subs={"TokenMixing", "SparseMoE"})    
class RankMixerBlock(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict):
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        dim_per_token = model_cfg[Const.HP].get("dim_per_token")
        all_dim = model_cfg[Const.HP].get("all_dim")
        num_experts = model_cfg[Const.HP].get("num_experts")
        inner_dim = model_cfg[Const.HP].get("inner_dim")
        ffn_layers = model_cfg[Const.HP].get("ffn_layers")
        T = model_cfg[Const.HP].get("T")
        k = model_cfg[Const.HP].get("k")
        k_multiplier = model_cfg[Const.HP].get("k_multiplier")
        dropout_p = model_cfg[Const.HP].get("dropout_p")
        target_activation_ratio = model_cfg[Const.HP].get("target_activation_ratio")
        sparseMOE = model_cfg[Const.HP].get("sparseMOE")
        top_k_number = model_cfg[Const.HP].get("top_k_number")
        
        self.model_cfg[Const.SUB_MODELS]["TokenMixing"][Const.HP] = {"dim_per_token": dim_per_token,
                                                                   "x_dim": all_dim,
                                                                   "k": k,
                                                                   "k_multiplier": k_multiplier}
        self.tokenmixing = self.init_sub_model("TokenMixing")
        
        self.model_cfg[Const.SUB_MODELS]["SparseMoE"][Const.HP] = {"num_experts_per_token": num_experts,
                                                                   "inner_dim": inner_dim,
                                                                   "num_tokens": T,
                                                                   "k": k,
                                                                   "num_layers_per_expert": ffn_layers,
                                                                   "bias": True,
                                                                   "dropout_p": dropout_p,
                                                                   "target_activation_ratio": target_activation_ratio,
                                                                   "sparseMOE":sparseMOE,
                                                                   "top_k_number":top_k_number
                                                                  }
        
        self.moe = self.init_sub_model("SparseMoE")
        
    def forward(self, x: torch.Tensor, loss_old: torch.Tensor, sparsity_old: torch.Tensor):
        with record_function("## tokenmixing ##"):
            x = self.tokenmixing(x)
        with record_function("## moe ##"):
            y, (loss, sparsity) = self.moe(x)
        return y, (loss + loss_old, sparsity + sparsity_old)
    

@ModelRegistry.register(req_subs={"RankMixingInput", "RankMixerBlock"})
class RankMixer(BaseModel):
    """
    根据论文复现RankMixer模型。
    """
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        feat_conf = common_hp["feature_conf"]
        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)
        # user_dim, item_dim = compute_user_item_feature_dims(feat_conf, model_conf)
        # all_dim = user_dim + item_dim
        all_dim = self._embedding_dim
        dim_per_token = model_cfg[Const.HP].get("dim_per_token", 428)
        # 如果不能被整除，找到最近的一个可以被整除的数，并且修改为新的dim_per_token
        if all_dim % dim_per_token != 0:
            new_dim_per_token = self.adjust_dim_fast(all_dim, dim_per_token)
            logging.warning("As the parameter dim_per_token %s is not a divisor of %s, we automatically modify it to the nearest divisor %s",
                           dim_per_token, all_dim, new_dim_per_token)
            dim_per_token = new_dim_per_token
            model_cfg[Const.HP]["dim_per_token"] = dim_per_token
        
        num_experts = model_cfg[Const.HP].get("num_experts", 10)
        k = model_cfg[Const.HP].get("k", 4)
        k_multiplier = model_cfg[Const.HP].get("k_multiplier", 1)
        ffn_layers = model_cfg[Const.HP].get("ffn_layers", 2)
        n_layers = model_cfg[Const.HP].get("n_layers", 2)
        dropout_p = model_cfg[Const.HP].get("dropout", 0.05)
        target_activation_ratio = model_cfg[Const.HP].get("target_activation_ratio", 1.0)
        sparseMOE = model_cfg[Const.HP].get("sparseMOE", True)
        top_k_number = model_cfg[Const.HP].get("top_k_number", 4)

        T = all_dim // dim_per_token
        inner_dim = int(T * k * k_multiplier)
        
        self.model_cfg[Const.SUB_MODELS]["RankMixingInput"][Const.HP] = {"dim_per_token": dim_per_token,
                                                                        "x_dim": all_dim,
                                                                        "k": k,
                                                                        "k_multiplier": k_multiplier}
        self.input_model = self.init_sub_model("RankMixingInput")
        
        self.model_cfg[Const.SUB_MODELS]["RankMixerBlock"][Const.HP] = {"dim_per_token": dim_per_token,
                                                                        "all_dim": all_dim,
                                                                        "num_experts": num_experts,
                                                                        "inner_dim": inner_dim,
                                                                        "ffn_layers": ffn_layers,
                                                                        "T": T,
                                                                        "k": k,
                                                                        "k_multiplier": k_multiplier,
                                                                        "dropout_p": dropout_p,
                                                                        "target_activation_ratio":target_activation_ratio,
                                                                        "sparseMOE":sparseMOE,
                                                                        "top_k_number": top_k_number
                                                                       } 
        self.rm_blocks = nn.Sequential(*[
                    self.init_sub_model("RankMixerBlock") 
                    for _ in range(n_layers)
                ])
        self.output_proj = nn.Linear(inner_dim, self._embedding_dim, bias=False)
        self.output_norm = RMSNormNPU(self._embedding_dim, Const.EPS)
        self.balance_loss_coef = 1e-8

    def adjust_dim_fast(self, all_dim, dim_per_token):
        divisors = [d for d in range(1, all_dim + 1) if all_dim % d == 0]
        return min(divisors, key=lambda x: abs(x - dim_per_token))
        
    def forward(self, 
                past_ids,
                num_rerank,
                model_inputs,
                user_feature_embs,
                item_feature_embs):
        results = []
        B = item_feature_embs.size(0)
        if user_feature_embs is not None:
            user_feature_embs = user_feature_embs.unsqueeze(1).expand(-1, num_rerank, -1)
            # 形状（B, N, D_u + D_i）
            x = torch.cat([user_feature_embs, item_feature_embs], dim=-1)
        else:
            x = item_feature_embs
        B, N, _ = x.size()
        x_batch = x.view(B * N, -1)
        rm_in = self.input_model(x_batch)
        l1_loss = 0.0
        sparsity = 0.0
        for i in range(len(self.rm_blocks)):
            block_i = self.rm_blocks[i]
            rm_in, (l1_loss, sparsity) = block_i(rm_in, l1_loss, sparsity)
        rm_out = rm_in.mean(dim=1)
        l1_loss = l1_loss / len(self.rm_blocks)
        sparsity = sparsity / len(self.rm_blocks)
        output = self.output_proj(rm_out)
        y = output.view(B, N, -1)
        y = self.output_norm(y)
        return {"deep_outputs": y, "deep_loss": l1_loss, "deep_sparsity": sparsity}
    
@ModelRegistry.register()
class NoDLRM(BaseModel):
    """
    用户-物品-评分输入特征预处理模块, 用于处理用户、物品和评分的特征。
    """

    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        
    def forward(self, 
                past_ids,
                num_rerank,
                model_inputs,
                user_feature_embs,
                item_feature_embs):
        B = user_feature_embs.size(0)
        N = num_rerank
        D = self._embedding_dim
        device = user_feature_embs.device
        return {"deep_outputs": torch.zeros((B, N, D), device=device), "deep_loss": 0.0, "deep_sparsity": 0.0}