import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from modeling.generic.sequential_v2.base_model import BaseModel
from modeling.generic.sequential_v2.transformers import RMSNormNPU
from modeling.generic.utils.constants import Const
from modeling.model_registry import ModelRegistry


NPU_ENABLE = os.environ.get("NPU_FLAG", "True") != "False"
TP_ALL_GATHER_INTO_TENSOR = os.environ.get("TP_ALL_GATHER_INTO_TENSOR", "1") != "0"
if NPU_ENABLE:
    try:
        import torch_npu  # type: ignore
    except Exception:
        torch_npu = None
else:
    torch_npu = None


def _get_npu_op(name: str):
    if torch_npu is not None:
        op = getattr(torch_npu, name, None)
        if op is not None:
            return op
    if hasattr(torch.ops, "npu"):
        return getattr(torch.ops.npu, name, None)
    return None


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _split_range(total: int, rank: int, size: int) -> Tuple[int, int]:
    base = total // size
    extra = total % size
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return start, end


def _pad_dim(x: torch.Tensor, dim: int, target_size: int) -> torch.Tensor:
    if dim < 0:
        dim += x.dim()
    current = x.size(dim)
    if current == target_size:
        return x
    if current > target_size:
        raise ValueError("target_size must be greater than or equal to current dim size")
    padding = [0] * (2 * x.dim())
    padding[2 * (x.dim() - dim - 1) + 1] = target_size - current
    return F.pad(x, tuple(padding), mode="constant", value=0.0)


def _all_gather_padded_dim(
    x: torch.Tensor,
    group,
    group_size: int,
    dim: int,
    sizes: Tuple[int, ...],
    max_size: int,
) -> torch.Tensor:
    padded = _pad_dim(x, dim, max_size).contiguous()
    if TP_ALL_GATHER_INTO_TENSOR and hasattr(dist, "all_gather_into_tensor"):
        send = padded.movedim(dim, 0).contiguous()
        recv_shape = (int(group_size) * int(max_size),) + tuple(send.shape[1:])
        recv = torch.empty(recv_shape, dtype=send.dtype, device=send.device)
        dist.all_gather_into_tensor(recv, send, group=group)
        return _assemble_gathered_buffer(recv, dim, sizes, max_size)

    gathered = [torch.empty_like(padded) for _ in range(int(group_size))]
    dist.all_gather(gathered, padded, group=group)
    gathered = [part.narrow(dim, 0, sizes[idx]) for idx, part in enumerate(gathered)]
    return torch.cat(gathered, dim=dim)


def _assemble_gathered_buffer(
    recv: torch.Tensor,
    dim: int,
    sizes: Tuple[int, ...],
    max_size: int,
) -> torch.Tensor:
    if all(size == int(max_size) for size in sizes):
        if dim == 0:
            return recv
        return recv.movedim(0, dim).contiguous()
    chunks = torch.split(recv, int(max_size), dim=0)
    return torch.cat(
        [chunk.narrow(0, 0, sizes[idx]).movedim(0, dim) for idx, chunk in enumerate(chunks)],
        dim=dim,
    )


def _reduce_scatter_padded_grad(ctx, grad_output: torch.Tensor) -> torch.Tensor:
    if all(size == ctx.max_size for size in ctx.sizes):
        reduce_input = grad_output.movedim(ctx.dim, 0).contiguous()
        reduce_output = torch.empty_like(reduce_input.narrow(0, 0, ctx.max_size))
    else:
        chunks = torch.split(grad_output, ctx.sizes, dim=ctx.dim)
        padded_chunks = [
            _pad_dim(chunk, ctx.dim, ctx.max_size).movedim(ctx.dim, 0).contiguous()
            for chunk in chunks
        ]
        reduce_input = torch.cat(padded_chunks, dim=0).contiguous()
        reduce_output = torch.empty_like(padded_chunks[ctx.group_rank])
    if hasattr(dist, "reduce_scatter_tensor"):
        work = dist.reduce_scatter_tensor(
            reduce_output,
            reduce_input,
            op=dist.ReduceOp.SUM,
            group=ctx.group,
            async_op=ctx.async_collectives,
        )
        if work is not None:
            work.wait()
    else:
        work = dist.all_reduce(
            reduce_input,
            op=dist.ReduceOp.SUM,
            group=ctx.group,
            async_op=ctx.async_collectives,
        )
        if work is not None:
            work.wait()
        start = ctx.group_rank * ctx.max_size
        reduce_output = reduce_input.narrow(0, start, ctx.max_size).contiguous()

    grad_input = reduce_output.movedim(0, ctx.dim)
    local_size = ctx.sizes[ctx.group_rank]
    if local_size != ctx.max_size:
        grad_input = grad_input.narrow(ctx.dim, 0, local_size)
    return grad_input


class _AllReduceMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, group_size):
        ctx.group = group
        ctx.group_size = int(group_size)
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out / float(ctx.group_size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        dist.all_reduce(grad_input, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_input / float(ctx.group_size), None, None


class _AllGatherDim(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, group_size, group_rank, dim, sizes, max_size, async_collectives):
        dim = int(dim)
        if dim < 0:
            dim += x.dim()
        sizes = tuple(int(size) for size in sizes)
        max_size = int(max_size)

        ctx.group = group
        ctx.group_size = int(group_size)
        ctx.group_rank = int(group_rank)
        ctx.dim = dim
        ctx.sizes = sizes
        ctx.max_size = max_size
        ctx.async_collectives = bool(async_collectives)

        return _all_gather_padded_dim(x, group, ctx.group_size, dim, sizes, max_size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _reduce_scatter_padded_grad(ctx, grad_output)
        return grad_input, None, None, None, None, None, None, None


class _AllGatherDimFromBuffer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, recv, group, group_size, group_rank, dim, sizes, max_size):
        dim = int(dim)
        if dim < 0:
            dim += x.dim()
        ctx.group = group
        ctx.group_size = int(group_size)
        ctx.group_rank = int(group_rank)
        ctx.dim = dim
        ctx.sizes = tuple(int(size) for size in sizes)
        ctx.max_size = int(max_size)
        ctx.async_collectives = True
        return _assemble_gathered_buffer(recv, dim, ctx.sizes, ctx.max_size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _reduce_scatter_padded_grad(ctx, grad_output)
        return grad_input, None, None, None, None, None, None, None


class _PendingAllGatherDim:
    def __init__(
        self,
        x: torch.Tensor,
        group,
        group_size: int,
        group_rank: int,
        dim: int,
        sizes: Tuple[int, ...],
        max_size: int,
        output_dtype: Optional[torch.dtype] = None,
    ) -> None:
        self.x = x
        self.group = group
        self.group_size = int(group_size)
        self.group_rank = int(group_rank)
        self.dim = int(dim) if dim >= 0 else int(dim) + x.dim()
        self.sizes = tuple(int(size) for size in sizes)
        self.max_size = int(max_size)
        self.output_dtype = output_dtype
        self.gathered = None
        self.send = None

        padded = _pad_dim(x, self.dim, self.max_size).contiguous()
        if TP_ALL_GATHER_INTO_TENSOR and hasattr(dist, "all_gather_into_tensor"):
            self.send = padded.movedim(self.dim, 0).contiguous()
            recv_shape = (self.group_size * self.max_size,) + tuple(self.send.shape[1:])
            self.recv = torch.empty(recv_shape, dtype=self.send.dtype, device=self.send.device)
            self.work = dist.all_gather_into_tensor(
                self.recv,
                self.send,
                group=self.group,
                async_op=True,
            )
        else:
            self.send = padded
            self.recv = None
            self.gathered = [torch.empty_like(padded) for _ in range(self.group_size)]
            self.work = dist.all_gather(
                self.gathered,
                self.send,
                group=self.group,
                async_op=True,
            )

    def wait(self) -> torch.Tensor:
        self.work.wait()
        recv = self.recv
        if recv is None:
            recv = torch.cat(
                [part.movedim(self.dim, 0) for part in self.gathered],
                dim=0,
            ).contiguous()
        if self.x.requires_grad:
            out = _AllGatherDimFromBuffer.apply(
                self.x,
                recv,
                self.group,
                self.group_size,
                self.group_rank,
                self.dim,
                self.sizes,
                self.max_size,
            )
        else:
            out = _assemble_gathered_buffer(recv, self.dim, self.sizes, self.max_size)
        if self.output_dtype is not None and out.dtype != self.output_dtype:
            out = out.to(dtype=self.output_dtype)
        return out


class TokenParallelRuntime:
    """Token Parallel runtime aligned with TokenMixer-Large data flow.

    The first RecSDK implementation keeps parameters materialized on every
    rank for checkpoint compatibility, while sharding expensive per-token
    computation across token/head dimensions and gathering activations back.
    """

    def __init__(
        self,
        enabled: bool,
        requested_size: int = 0,
        reduce_aux_loss: bool = False,
        comm_overlap: bool = False,
        comm_dtype: str = "float32",
    ) -> None:
        self.enabled = False
        self.group = None
        self.group_size = 1
        self.group_rank = 0
        self.global_rank = 0
        self.reduce_aux_loss = bool(reduce_aux_loss)
        self.comm_overlap = bool(comm_overlap)
        self.comm_dtype = str(comm_dtype).strip().lower()
        if self.comm_dtype not in ("float32", "fp32", "bfloat16", "bf16", "float16", "fp16"):
            raise ValueError("Unsupported Token Parallel communication dtype: %s" % comm_dtype)
        if not enabled or not _dist_ready():
            return

        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        tp_size = int(requested_size) if requested_size else world_size
        if tp_size <= 1:
            return
        if world_size % tp_size != 0:
            logging.warning(
                "Token Parallel disabled: world_size=%s is not divisible by token_parallel_size=%s",
                world_size,
                tp_size,
            )
            return

        selected_group = None
        for start in range(0, world_size, tp_size):
            ranks = list(range(start, start + tp_size))
            group = dist.new_group(ranks=ranks)
            if global_rank in ranks:
                selected_group = group
                self.group_rank = global_rank - start

        self.enabled = selected_group is not None
        self.group = selected_group
        self.group_size = tp_size if self.enabled else 1
        self.global_rank = global_rank
        if self.enabled and self.comm_overlap and self.global_rank == 0:
            logging.info(
                "Token Parallel full-tensor async communication overlap enabled, comm_dtype=%s",
                self.comm_dtype,
            )

    def shard_range(self, total: int) -> Tuple[int, int]:
        return _split_range(total, self.group_rank, self.group_size)

    def _prepare_comm_input(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.dtype]]:
        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
        }
        target_dtype = dtype_map[self.comm_dtype]
        target_element_size = 4 if target_dtype == torch.float32 else 2
        if not x.is_floating_point() or x.element_size() <= target_element_size:
            return x, None
        return x.to(dtype=target_dtype), x.dtype

    def all_gather_dim(
        self,
        x: torch.Tensor,
        dim: int,
        total_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[int], int]:
        if not self.enabled:
            return x, [x.size(dim)], 0

        if total_size is None:
            local_size = torch.tensor([x.size(dim)], device=x.device, dtype=torch.int64)
            gathered_sizes = [torch.empty_like(local_size) for _ in range(self.group_size)]
            dist.all_gather(gathered_sizes, local_size, group=self.group)
            sizes = [int(size.item()) for size in gathered_sizes]
        else:
            sizes = [
                _split_range(int(total_size), rank, self.group_size)[1]
                - _split_range(int(total_size), rank, self.group_size)[0]
                for rank in range(self.group_size)
            ]
        offsets = [0]
        for size in sizes[:-1]:
            offsets.append(offsets[-1] + size)

        max_size = max(sizes)
        comm_x, output_dtype = self._prepare_comm_input(x)
        if comm_x.requires_grad:
            gathered = _AllGatherDim.apply(
                comm_x,
                self.group,
                self.group_size,
                self.group_rank,
                dim,
                tuple(sizes),
                max_size,
                self.comm_overlap,
            )
        else:
            gathered = _all_gather_padded_dim(
                comm_x,
                self.group,
                self.group_size,
                dim,
                tuple(sizes),
                max_size,
            )
        if output_dtype is not None and gathered.dtype != output_dtype:
            gathered = gathered.to(dtype=output_dtype)
        return gathered, sizes, offsets[self.group_rank]

    def begin_all_gather_dim(
        self,
        x: torch.Tensor,
        dim: int,
        total_size: int,
    ) -> _PendingAllGatherDim:
        if not self.enabled:
            raise RuntimeError("begin_all_gather_dim requires Token Parallel to be enabled")
        sizes = tuple(
            _split_range(int(total_size), rank, self.group_size)[1]
            - _split_range(int(total_size), rank, self.group_size)[0]
            for rank in range(self.group_size)
        )
        comm_x, output_dtype = self._prepare_comm_input(x)
        return _PendingAllGatherDim(
            comm_x,
            self.group,
            self.group_size,
            self.group_rank,
            dim,
            sizes,
            max(sizes),
            output_dtype=output_dtype,
        )

    def should_overlap(self, x: torch.Tensor) -> bool:
        if not self.enabled or not self.comm_overlap:
            return False
        if x.device.type not in ("npu", "cuda"):
            return False
        return True

    def all_reduce_mean(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return _AllReduceMean.apply(x, self.group, self.group_size)


def _make_norm(dim: int, eps: float, use_rmsnorm: bool) -> nn.Module:
    if use_rmsnorm:
        return RMSNormNPU(dim, eps)
    return nn.LayerNorm(dim, eps=eps)


class PerTokenSwiGLU(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        hidden_multiplier: float,
        dropout_p: float,
        bias: bool,
        down_init_scale: float,
        npu_bmm: bool = True,
        fused_gate_up: bool = True,
        use_npu_swiglu: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(round(dim * hidden_multiplier)))
        self.num_tokens = num_tokens
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.npu_bmm = npu_bmm
        self.fused_gate_up = fused_gate_up
        self.use_npu_swiglu = use_npu_swiglu
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.gate_up_weight = nn.Parameter(torch.empty(num_tokens, dim, hidden_dim * 2))
        self.down_weight = nn.Parameter(torch.empty(num_tokens, hidden_dim, dim))
        if bias:
            self.gate_up_bias = nn.Parameter(torch.empty(num_tokens, hidden_dim * 2))
            self.down_bias = nn.Parameter(torch.empty(num_tokens, dim))
        else:
            self.register_parameter("gate_up_bias", None)
            self.register_parameter("down_bias", None)
        self.down_init_scale = down_init_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        gate_weight, up_weight = self.gate_up_weight.split(self.hidden_dim, dim=-1)
        nn.init.xavier_normal_(gate_weight)
        nn.init.xavier_normal_(up_weight)
        nn.init.xavier_normal_(self.down_weight)
        if self.down_init_scale != 1.0:
            with torch.no_grad():
                self.down_weight.mul_(self.down_init_scale)
        if self.gate_up_bias is not None:
            nn.init.zeros_(self.gate_up_bias)
            nn.init.zeros_(self.down_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self._gate_up(x)
        if self.gate_up_bias is not None:
            gate_up = gate_up + self.gate_up_bias
        y = self._swiglu(gate_up)
        y = self.dropout(y)
        y = self._per_token_linear(y, self.down_weight)
        if self.down_bias is not None:
            y = y + self.down_bias
        return self.dropout(y)

    def _forward_with_params(
        self,
        x: torch.Tensor,
        gate_up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        gate_up_bias: Optional[torch.Tensor],
        down_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.npu_bmm:
            gate_up = torch.bmm(x.transpose(0, 1), gate_up_weight).transpose(0, 1)
        else:
            gate_up = torch.einsum("btd,tdh->bth", x, gate_up_weight)
        if gate_up_bias is not None:
            gate_up = gate_up + gate_up_bias
        y = self._swiglu(gate_up)
        y = self.dropout(y)
        if self.npu_bmm:
            y = torch.bmm(y.transpose(0, 1), down_weight).transpose(0, 1)
        else:
            y = torch.einsum("bth,thd->btd", y, down_weight)
        if down_bias is not None:
            y = y + down_bias
        return self.dropout(y)

    def forward_shard(self, x: torch.Tensor, token_start: int) -> torch.Tensor:
        local_tokens = x.size(1)
        if local_tokens == 0:
            return x.new_zeros(x.shape)
        token_end = token_start + local_tokens
        gate_up_bias = self.gate_up_bias[token_start:token_end] if self.gate_up_bias is not None else None
        down_bias = self.down_bias[token_start:token_end] if self.down_bias is not None else None
        return self._forward_with_params(
            x,
            self.gate_up_weight[token_start:token_end],
            self.down_weight[token_start:token_end],
            gate_up_bias,
            down_bias,
        )

    def _gate_up(self, x: torch.Tensor) -> torch.Tensor:
        if self.npu_bmm:
            xt = x.transpose(0, 1)
            if self.fused_gate_up:
                return torch.bmm(xt, self.gate_up_weight).transpose(0, 1)
            gate_weight, up_weight = self.gate_up_weight.split(self.hidden_dim, dim=-1)
            gate = torch.bmm(xt, gate_weight).transpose(0, 1)
            up = torch.bmm(xt, up_weight).transpose(0, 1)
            return torch.cat([gate, up], dim=-1)
        gate_weight, up_weight = self.gate_up_weight.split(self.hidden_dim, dim=-1)
        gate = torch.einsum("btd,tdh->bth", x, gate_weight)
        up = torch.einsum("btd,tdh->bth", x, up_weight)
        return torch.cat([gate, up], dim=-1)

    def _swiglu(self, gate_up: torch.Tensor) -> torch.Tensor:
        npu_swiglu = None
        if self.use_npu_swiglu and gate_up.device.type == "npu":
            npu_swiglu = _get_npu_op("npu_swiglu")
        if npu_swiglu is not None:
            return npu_swiglu(gate_up, dim=-1)
        gate, up = gate_up.split(self.hidden_dim, dim=-1)
        return F.silu(gate) * up

    def _per_token_linear(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        if self.npu_bmm:
            return torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)
        return torch.einsum("bth,thd->btd", x, weight)

    def forward_selected(
        self,
        x: torch.Tensor,
        flat_indices: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_indices = torch.arange(self.num_tokens, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        flat_token_indices = token_indices.reshape(-1).index_select(0, flat_indices)
        selected = x.reshape(-1, self.dim).index_select(0, flat_indices).unsqueeze(1)
        gate_up_weight = self.gate_up_weight.index_select(0, flat_token_indices)
        gate_up = torch.bmm(selected, gate_up_weight).squeeze(1)
        if self.gate_up_bias is not None:
            gate_up = gate_up + self.gate_up_bias.index_select(0, flat_token_indices)
        y = self._swiglu(gate_up)
        y = self.dropout(y)
        down_weight = self.down_weight.index_select(0, flat_token_indices)
        y = torch.bmm(y.unsqueeze(1), down_weight).squeeze(1)
        if self.down_bias is not None:
            y = y + self.down_bias.index_select(0, flat_token_indices)
        if weights is not None:
            y = y * weights.unsqueeze(-1)
        return self.dropout(y)


class LegacyPerTokenFFN(nn.Module):
    """TokenMixer-Large fallback for ablation only: per-token FFN without SwiGLU."""

    def __init__(
        self,
        num_tokens: int,
        dim: int,
        hidden_multiplier: float,
        dropout_p: float,
        bias: bool,
        down_init_scale: float,
        npu_bmm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = max(1, int(round(dim * hidden_multiplier)))
        self.num_tokens = num_tokens
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.npu_bmm = npu_bmm
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()
        self.up_weight = nn.Parameter(torch.empty(num_tokens, dim, hidden_dim))
        self.down_weight = nn.Parameter(torch.empty(num_tokens, hidden_dim, dim))
        if bias:
            self.up_bias = nn.Parameter(torch.empty(num_tokens, hidden_dim))
            self.down_bias = nn.Parameter(torch.empty(num_tokens, dim))
        else:
            self.register_parameter("up_bias", None)
            self.register_parameter("down_bias", None)
        self.down_init_scale = down_init_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.up_weight)
        nn.init.xavier_normal_(self.down_weight)
        if self.down_init_scale != 1.0:
            with torch.no_grad():
                self.down_weight.mul_(self.down_init_scale)
        if self.up_bias is not None:
            nn.init.zeros_(self.up_bias)
            nn.init.zeros_(self.down_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.npu_bmm:
            y = torch.bmm(x.transpose(0, 1), self.up_weight).transpose(0, 1)
        else:
            y = torch.einsum("btd,tdh->bth", x, self.up_weight)
        if self.up_bias is not None:
            y = y + self.up_bias
        y = F.gelu(y)
        y = self.dropout(y)
        if self.npu_bmm:
            y = torch.bmm(y.transpose(0, 1), self.down_weight).transpose(0, 1)
        else:
            y = torch.einsum("bth,thd->btd", y, self.down_weight)
        if self.down_bias is not None:
            y = y + self.down_bias
        return self.dropout(y)

    def _forward_with_params(
        self,
        x: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        up_bias: Optional[torch.Tensor],
        down_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.npu_bmm:
            y = torch.bmm(x.transpose(0, 1), up_weight).transpose(0, 1)
        else:
            y = torch.einsum("btd,tdh->bth", x, up_weight)
        if up_bias is not None:
            y = y + up_bias
        y = F.gelu(y)
        y = self.dropout(y)
        if self.npu_bmm:
            y = torch.bmm(y.transpose(0, 1), down_weight).transpose(0, 1)
        else:
            y = torch.einsum("bth,thd->btd", y, down_weight)
        if down_bias is not None:
            y = y + down_bias
        return self.dropout(y)

    def forward_shard(self, x: torch.Tensor, token_start: int) -> torch.Tensor:
        local_tokens = x.size(1)
        if local_tokens == 0:
            return x.new_zeros(x.shape)
        token_end = token_start + local_tokens
        up_bias = self.up_bias[token_start:token_end] if self.up_bias is not None else None
        down_bias = self.down_bias[token_start:token_end] if self.down_bias is not None else None
        return self._forward_with_params(
            x,
            self.up_weight[token_start:token_end],
            self.down_weight[token_start:token_end],
            up_bias,
            down_bias,
        )


class SparsePerTokenMoE(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        num_experts: int,
        top_k: int,
        hidden_multiplier: float,
        dropout_p: float,
        bias: bool,
        down_init_scale: float,
        gate_scale_alpha: Optional[float] = None,
        use_shared_expert: bool = True,
        npu_bmm: bool = True,
        fused_gate_up: bool = True,
        use_npu_swiglu: bool = True,
        sparse_top1_dispatch: bool = True,
        fast_top1_gating: bool = True,
        use_npu_grouped_moe: bool = True,
        use_npu_grouped_moe_bias_fusion: bool = True,
        fallback_on_npu_grouped_moe_failure: bool = True,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.hidden_dim = max(1, int(round(dim * hidden_multiplier)))
        self.dropout_p = dropout_p
        self.num_experts = max(1, num_experts)
        self.top_k = max(1, min(top_k, self.num_experts))
        self.npu_bmm = npu_bmm
        self.use_npu_swiglu = use_npu_swiglu
        self.sparse_top1_dispatch = sparse_top1_dispatch
        self.fast_top1_gating = fast_top1_gating
        self.use_npu_grouped_moe = use_npu_grouped_moe
        self.use_npu_grouped_moe_bias_fusion = use_npu_grouped_moe_bias_fusion
        self._npu_grouped_moe_bias_fusion_disabled = False
        self.fallback_on_npu_grouped_moe_failure = fallback_on_npu_grouped_moe_failure
        self._npu_grouped_moe_disabled = False
        self.bias = bias
        self.down_init_scale = down_init_scale
        default_scale = float(self.num_experts) / float(self.top_k)
        self.gate_scale_alpha = gate_scale_alpha if gate_scale_alpha is not None else default_scale
        self.router = nn.Parameter(torch.empty(num_tokens, dim, self.num_experts))
        self.gate_up_weight = nn.Parameter(
            torch.empty(self.num_tokens * self.num_experts, self.dim, self.hidden_dim * 2)
        )
        self.down_weight = nn.Parameter(
            torch.empty(self.num_tokens * self.num_experts, self.hidden_dim, self.dim)
        )
        if bias:
            self.gate_up_bias = nn.Parameter(torch.empty(self.num_tokens * self.num_experts, self.hidden_dim * 2))
            self.down_bias = nn.Parameter(torch.empty(self.num_tokens * self.num_experts, self.dim))
        else:
            self.register_parameter("gate_up_bias", None)
            self.register_parameter("down_bias", None)
        self.shared_expert = (
            PerTokenSwiGLU(
                num_tokens,
                dim,
                hidden_multiplier,
                dropout_p,
                bias,
                down_init_scale,
                npu_bmm=npu_bmm,
                fused_gate_up=fused_gate_up,
                use_npu_swiglu=use_npu_swiglu,
            )
            if use_shared_expert
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_normal_(self.router)
        gate_weight, up_weight = self.gate_up_weight.split(self.hidden_dim, dim=-1)
        nn.init.xavier_normal_(gate_weight)
        nn.init.xavier_normal_(up_weight)
        nn.init.xavier_normal_(self.down_weight)
        if self.down_init_scale != 1.0:
            with torch.no_grad():
                self.down_weight.mul_(self.down_init_scale)
        if self.gate_up_bias is not None:
            nn.init.zeros_(self.gate_up_bias)
            nn.init.zeros_(self.down_bias)

    def forward(self, x: torch.Tensor, compute_aux_loss: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.npu_bmm:
            logits = torch.bmm(x.transpose(0, 1), self.router).transpose(0, 1)
        else:
            logits = torch.einsum("btd,tde->bte", x, self.router)
        if self.fast_top1_gating and self.top_k == 1:
            topk_indices = torch.argmax(logits, dim=-1, keepdim=True)
            topk_gates = logits.new_full(topk_indices.shape, float(self.gate_scale_alpha))
        else:
            topk_scores, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
            topk_gates = F.softmax(topk_scores, dim=-1) * self.gate_scale_alpha

        if self._can_use_npu_grouped_moe(x):
            try:
                out = self._forward_npu_grouped_moe(x, topk_indices, topk_gates)
            except Exception as exc:
                self._npu_grouped_moe_disabled = True
                if not self.fallback_on_npu_grouped_moe_failure:
                    raise
                logging.warning("NPU grouped MoE fallback to PyTorch path: %s", exc)
                out = self._forward_fallback_moe(x, topk_indices, topk_gates)
        else:
            out = self._forward_fallback_moe(x, topk_indices, topk_gates)
        if self.shared_expert is not None:
            out = out + self.shared_expert(x)

        aux_loss = self.compute_aux_loss_from_logits(logits, x) if compute_aux_loss else x.float().new_tensor(0.0)
        active_ratio = x.new_tensor(float(self.top_k) / float(self.num_experts))
        return out, aux_loss, active_ratio

    def forward_shard(
        self,
        x: torch.Tensor,
        token_start: int,
        compute_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, logits, active_ratio = self.forward_shard_deferred_aux(x, token_start)
        aux_loss = (
            self.compute_aux_loss_from_logits(logits, x)
            if compute_aux_loss
            else x.float().new_tensor(0.0)
        )
        return out, aux_loss, active_ratio

    def forward_shard_deferred_aux(
        self,
        x: torch.Tensor,
        token_start: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        local_tokens = x.size(1)
        if local_tokens == 0:
            return x.new_zeros(x.shape), None, x.new_tensor(float(self.top_k) / float(self.num_experts))

        token_end = token_start + local_tokens
        local_router = self.router[token_start:token_end]
        if self.npu_bmm:
            logits = torch.bmm(x.transpose(0, 1), local_router).transpose(0, 1)
        else:
            logits = torch.einsum("btd,tde->bte", x, local_router)
        if self.fast_top1_gating and self.top_k == 1:
            topk_indices = torch.argmax(logits, dim=-1, keepdim=True)
            topk_gates = logits.new_full(topk_indices.shape, float(self.gate_scale_alpha))
        else:
            topk_scores, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
            topk_gates = F.softmax(topk_scores, dim=-1) * self.gate_scale_alpha

        if self._can_use_npu_grouped_moe(x):
            try:
                out = self._forward_npu_grouped_moe(x, topk_indices, topk_gates, token_start=token_start)
            except Exception as exc:
                self._npu_grouped_moe_disabled = True
                if not self.fallback_on_npu_grouped_moe_failure:
                    raise
                logging.warning("NPU grouped MoE shard fallback to PyTorch path: %s", exc)
                out = self._forward_fallback_moe(x, topk_indices, topk_gates, token_start=token_start)
        else:
            out = self._forward_fallback_moe(x, topk_indices, topk_gates, token_start=token_start)
        if self.shared_expert is not None:
            out = out + self.shared_expert.forward_shard(x, token_start)

        active_ratio = x.new_tensor(float(self.top_k) / float(self.num_experts))
        return out, logits, active_ratio

    def compute_aux_loss_from_logits(
        self,
        logits: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if logits is None:
            return reference.float().new_tensor(0.0)
        probs = F.softmax(logits.float(), dim=-1)
        mean_prob = probs.mean(dim=(0, 1))
        return mean_prob.pow(2).sum() * float(self.num_experts)

    def _forward_fallback_moe(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_gates: torch.Tensor,
        token_start: int = 0,
    ) -> torch.Tensor:
        if self.sparse_top1_dispatch and self.top_k == 1:
            return self._forward_top1_sparse(x, topk_indices.squeeze(-1), topk_gates.squeeze(-1), token_start=token_start)
        return self._forward_dense_expert_loop(x, topk_indices, topk_gates, token_start=token_start)

    def _can_use_npu_grouped_moe(self, x: torch.Tensor) -> bool:
        if not self.use_npu_grouped_moe or self._npu_grouped_moe_disabled:
            return False
        if x.device.type != "npu":
            return False
        required_ops = (
            "npu_moe_token_permute",
            "npu_grouped_matmul",
            "npu_moe_token_unpermute",
        )
        return all(_get_npu_op(name) is not None for name in required_ops)

    def _grouped_gate_up_weight(self) -> torch.Tensor:
        return self.gate_up_weight

    def _grouped_down_weight(self) -> torch.Tensor:
        return self.down_weight

    def _grouped_gate_up_bias(self) -> Optional[torch.Tensor]:
        return self.gate_up_bias

    def _grouped_down_bias(self) -> Optional[torch.Tensor]:
        return self.down_bias

    def _npu_grouped_matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        group_list: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, bool]:
        npu_grouped_matmul = _get_npu_op("npu_grouped_matmul")
        if npu_grouped_matmul is None:
            raise RuntimeError("npu_grouped_matmul is not available")
        kwargs = {
            "x": [x],
            "weight": [weight],
            "group_list": group_list,
            "split_item": 2,
            "group_list_type": 0,
            "group_type": 0,
            "output_dtype": x.dtype,
        }
        fuse_bias = (
            bias is not None
            and self.use_npu_grouped_moe_bias_fusion
            and not self._npu_grouped_moe_bias_fusion_disabled
        )
        if fuse_bias:
            try:
                return npu_grouped_matmul(bias=[bias], **kwargs)[0], True
            except (RuntimeError, TypeError) as exc:
                self._npu_grouped_moe_bias_fusion_disabled = True
                logging.warning("NPU grouped MoE bias fusion disabled: %s", exc)
        return npu_grouped_matmul(**kwargs)[0], False

    def _swiglu(self, gate_up: torch.Tensor) -> torch.Tensor:
        npu_swiglu = _get_npu_op("npu_swiglu") if self.use_npu_swiglu and gate_up.device.type == "npu" else None
        if npu_swiglu is not None:
            return npu_swiglu(gate_up, dim=-1)
        gate, up = gate_up.split(self.hidden_dim, dim=-1)
        return F.silu(gate) * up

    def _forward_npu_grouped_moe(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_gates: torch.Tensor,
        token_start: int = 0,
    ) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape
        top_k = topk_indices.size(-1)
        num_groups = self.num_tokens * self.num_experts
        num_out_tokens = batch_size * num_tokens * top_k

        token_offsets = torch.arange(
            token_start,
            token_start + num_tokens,
            device=x.device,
            dtype=topk_indices.dtype,
        ).view(1, num_tokens, 1)
        group_indices = topk_indices + token_offsets * self.num_experts
        flat_group_indices = group_indices.reshape(batch_size * num_tokens, top_k)
        flat_group_ids = flat_group_indices.reshape(-1).to(torch.long)

        group_counts = F.one_hot(flat_group_ids, num_classes=num_groups).sum(dim=0).to(torch.int64)
        group_list = group_counts.cumsum(dim=0)
        npu_moe_token_permute = _get_npu_op("npu_moe_token_permute")
        npu_moe_token_unpermute = _get_npu_op("npu_moe_token_unpermute")
        if npu_moe_token_permute is None or npu_moe_token_unpermute is None:
            raise RuntimeError("NPU MoE token permute/unpermute ops are not available")

        tokens = x.reshape(batch_size * num_tokens, dim)
        expanded_x, row_index = npu_moe_token_permute(
            tokens=tokens,
            indices=flat_group_indices,
            num_out_tokens=num_out_tokens,
        )

        gate_up_bias = self._grouped_gate_up_bias()
        gate_up, gate_up_bias_fused = self._npu_grouped_matmul(
            expanded_x,
            self._grouped_gate_up_weight(),
            group_list,
            bias=gate_up_bias,
        )
        sorted_group_ids = None
        if gate_up_bias is not None and not gate_up_bias_fused:
            sorted_group_ids = torch.repeat_interleave(
                torch.arange(num_groups, device=x.device, dtype=torch.long),
                group_counts.to(torch.long),
            )
            gate_up = gate_up + gate_up_bias.index_select(0, sorted_group_ids)

        hidden = self._swiglu(gate_up)
        hidden = F.dropout(hidden, p=self.dropout_p, training=self.training)

        down_bias = self._grouped_down_bias()
        expert_out, down_bias_fused = self._npu_grouped_matmul(
            hidden,
            self._grouped_down_weight(),
            group_list,
            bias=down_bias,
        )
        if down_bias is not None and not down_bias_fused:
            if sorted_group_ids is None:
                sorted_group_ids = torch.repeat_interleave(
                    torch.arange(num_groups, device=x.device, dtype=torch.long),
                    group_counts.to(torch.long),
                )
            expert_out = expert_out + down_bias.index_select(0, sorted_group_ids)
        expert_out = F.dropout(expert_out, p=self.dropout_p, training=self.training)

        router_weights = topk_gates.reshape(-1, 1)
        combined = npu_moe_token_unpermute(expert_out, row_index, router_weights)
        if combined.numel() == batch_size * num_tokens * dim:
            return combined.view(batch_size, num_tokens, dim)
        return combined.view(batch_size, num_tokens, top_k, dim).sum(dim=2)

    def _expert_group_ids(self, num_tokens: int, device: torch.device, expert_id: int, token_start: int) -> torch.Tensor:
        token_ids = torch.arange(token_start, token_start + num_tokens, device=device, dtype=torch.long)
        return token_ids * self.num_experts + int(expert_id)

    def _forward_expert_tokens(self, x: torch.Tensor, expert_id: int, token_start: int) -> torch.Tensor:
        group_ids = self._expert_group_ids(x.size(1), x.device, expert_id, token_start)
        gate_up_weight = self.gate_up_weight.index_select(0, group_ids)
        if self.npu_bmm:
            gate_up = torch.bmm(x.transpose(0, 1), gate_up_weight).transpose(0, 1)
        else:
            gate_up = torch.einsum("btd,tdh->bth", x, gate_up_weight)
        if self.gate_up_bias is not None:
            gate_up = gate_up + self.gate_up_bias.index_select(0, group_ids)
        hidden = self._swiglu(gate_up)
        hidden = F.dropout(hidden, p=self.dropout_p, training=self.training)
        down_weight = self.down_weight.index_select(0, group_ids)
        if self.npu_bmm:
            out = torch.bmm(hidden.transpose(0, 1), down_weight).transpose(0, 1)
        else:
            out = torch.einsum("bth,thd->btd", hidden, down_weight)
        if self.down_bias is not None:
            out = out + self.down_bias.index_select(0, group_ids)
        return F.dropout(out, p=self.dropout_p, training=self.training)

    def _forward_selected_tokens(
        self,
        x: torch.Tensor,
        flat_indices: torch.Tensor,
        expert_id: int,
        token_start: int,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, local_tokens, _ = x.shape
        token_indices = (
            torch.arange(token_start, token_start + local_tokens, device=x.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        flat_token_indices = token_indices.reshape(-1).index_select(0, flat_indices)
        group_ids = flat_token_indices * self.num_experts + int(expert_id)
        selected = x.reshape(-1, self.dim).index_select(0, flat_indices).unsqueeze(1)
        gate_up_weight = self.gate_up_weight.index_select(0, group_ids)
        gate_up = torch.bmm(selected, gate_up_weight).squeeze(1)
        if self.gate_up_bias is not None:
            gate_up = gate_up + self.gate_up_bias.index_select(0, group_ids)
        hidden = self._swiglu(gate_up)
        hidden = F.dropout(hidden, p=self.dropout_p, training=self.training)
        down_weight = self.down_weight.index_select(0, group_ids)
        out = torch.bmm(hidden.unsqueeze(1), down_weight).squeeze(1)
        if self.down_bias is not None:
            out = out + self.down_bias.index_select(0, group_ids)
        if weights is not None:
            out = out * weights.unsqueeze(-1)
        return F.dropout(out, p=self.dropout_p, training=self.training)

    def _forward_dense_expert_loop(
        self,
        x: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_gates: torch.Tensor,
        token_start: int = 0,
    ) -> torch.Tensor:
        out = x.new_zeros(x.shape)
        for expert_id in range(self.num_experts):
            expert_mask = topk_indices == expert_id
            if not expert_mask.any():
                continue
            weights = (topk_gates * expert_mask.to(dtype=topk_gates.dtype)).sum(dim=-1)
            out = out + self._forward_expert_tokens(x, expert_id, token_start) * weights.unsqueeze(-1)
        return out

    def _forward_top1_sparse(
        self,
        x: torch.Tensor,
        top1_indices: torch.Tensor,
        top1_gates: torch.Tensor,
        token_start: int = 0,
    ) -> torch.Tensor:
        flat_expert = top1_indices.reshape(-1)
        flat_gates = top1_gates.reshape(-1)
        out_flat = x.new_zeros(x.numel() // self.dim, self.dim)
        for expert_id in range(self.num_experts):
            flat_indices = torch.nonzero(flat_expert == expert_id, as_tuple=False).flatten()
            if flat_indices.numel() == 0:
                continue
            selected_weights = flat_gates.index_select(0, flat_indices)
            selected_out = self._forward_selected_tokens(x, flat_indices, expert_id, token_start, selected_weights)
            out_flat.index_add_(0, flat_indices, selected_out)
        return out_flat.view_as(x)


class TokenMixerLargeBlock(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        dim: int,
        num_heads: int,
        hidden_multiplier: float,
        dropout_p: float,
        bias: bool,
        down_init_scale: float,
        use_rmsnorm: bool,
        use_pswiglu: bool,
        use_mixing_reverting: bool,
        use_aux_loss: bool,
        activation_aux_scale: float,
        moe_type: str,
        num_experts: int,
        moe_topk: int,
        gate_scale_alpha: Optional[float],
        moe_shared_expert: bool,
        norm_eps: float,
        npu_bmm: bool,
        fused_gate_up: bool,
        use_npu_swiglu: bool,
        sparse_top1_dispatch: bool,
        fast_top1_gating: bool,
        use_npu_grouped_moe: bool,
        use_npu_grouped_moe_bias_fusion: bool,
        fallback_on_npu_grouped_moe_failure: bool,
        use_fused_add_rmsnorm: bool,
        validate_static_shapes: bool,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim must be divisible by num_heads, got dim={dim}, num_heads={num_heads}"
            )
        self.num_tokens = num_tokens
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_mixing_reverting = use_mixing_reverting
        self.use_aux_loss = use_aux_loss
        self.activation_aux_scale = activation_aux_scale
        self.use_fused_add_rmsnorm = use_fused_add_rmsnorm
        self.validate_static_shapes = validate_static_shapes
        # The paper applies dropout to each residual branch after its MoE
        # update.  The per-token FFN/MoE may also have its own internal
        # dropout; this module-level dropout is a separate residual-path op.
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        self.mixed_tokens = num_heads if use_mixing_reverting else num_tokens
        self.mixed_dim = num_tokens * self.head_dim if use_mixing_reverting else dim
        self.mixed_norm = _make_norm(self.mixed_dim, norm_eps, use_rmsnorm)
        self.mixed_ffn = self._make_ffn(
            self.mixed_tokens,
            self.mixed_dim,
            hidden_multiplier,
            0.0,
            bias,
            down_init_scale,
            use_pswiglu,
            moe_type,
            num_experts,
            moe_topk,
            gate_scale_alpha,
            moe_shared_expert,
            npu_bmm,
            fused_gate_up,
            use_npu_swiglu,
            sparse_top1_dispatch,
            fast_top1_gating,
            use_npu_grouped_moe,
            use_npu_grouped_moe_bias_fusion,
            fallback_on_npu_grouped_moe_failure,
        )
        self.original_norm = _make_norm(dim, norm_eps, use_rmsnorm)
        self.original_ffn = self._make_ffn(
            num_tokens,
            dim,
            hidden_multiplier,
            0.0,
            bias,
            down_init_scale,
            use_pswiglu,
            moe_type,
            num_experts,
            moe_topk,
            gate_scale_alpha,
            moe_shared_expert,
            npu_bmm,
            fused_gate_up,
            use_npu_swiglu,
            sparse_top1_dispatch,
            fast_top1_gating,
            use_npu_grouped_moe,
            use_npu_grouped_moe_bias_fusion,
            fallback_on_npu_grouped_moe_failure,
        )

    @staticmethod
    def _make_ffn(
        num_tokens: int,
        dim: int,
        hidden_multiplier: float,
        dropout_p: float,
        bias: bool,
        down_init_scale: float,
        use_pswiglu: bool,
        moe_type: str,
        num_experts: int,
        moe_topk: int,
        gate_scale_alpha: Optional[float],
        moe_shared_expert: bool,
        npu_bmm: bool,
        fused_gate_up: bool,
        use_npu_swiglu: bool,
        sparse_top1_dispatch: bool,
        fast_top1_gating: bool,
        use_npu_grouped_moe: bool,
        use_npu_grouped_moe_bias_fusion: bool,
        fallback_on_npu_grouped_moe_failure: bool,
    ) -> nn.Module:
        if moe_type in {"spmoe", "sparse_pertoken", "sparse-pertoken"}:
            return SparsePerTokenMoE(
                num_tokens,
                dim,
                num_experts,
                moe_topk,
                hidden_multiplier,
                dropout_p,
                bias,
                down_init_scale,
                gate_scale_alpha,
                moe_shared_expert,
                npu_bmm,
                fused_gate_up,
                use_npu_swiglu,
                sparse_top1_dispatch,
                fast_top1_gating,
                use_npu_grouped_moe,
                use_npu_grouped_moe_bias_fusion,
                fallback_on_npu_grouped_moe_failure,
            )
        if use_pswiglu:
            return PerTokenSwiGLU(
                num_tokens,
                dim,
                hidden_multiplier,
                dropout_p,
                bias,
                down_init_scale,
                npu_bmm,
                fused_gate_up,
                use_npu_swiglu,
            )
        return LegacyPerTokenFFN(num_tokens, dim, hidden_multiplier, dropout_p, bias, down_init_scale, npu_bmm)

    def mix(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return (
            x.reshape(batch_size, self.num_tokens, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size, self.num_heads, self.mixed_dim)
        )

    def revert(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return (
            x.reshape(batch_size, self.num_heads, self.num_tokens, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(batch_size, self.num_tokens, self.dim)
        )

    @staticmethod
    def _run_ffn(
        module: nn.Module,
        x: torch.Tensor,
        compute_aux_loss: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(module, SparsePerTokenMoE):
            return module(x, compute_aux_loss=compute_aux_loss)
        out = module(x)
        return out, out.float().new_tensor(0.0), out.new_tensor(1.0)

    @staticmethod
    def _run_ffn_shard(
        module: nn.Module,
        x: torch.Tensor,
        compute_aux_loss: bool,
        token_start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(module, SparsePerTokenMoE):
            return module.forward_shard(x, token_start, compute_aux_loss=compute_aux_loss)
        if hasattr(module, "forward_shard"):
            out = module.forward_shard(x, token_start)
        else:
            out = module(x)
        return out, out.float().new_tensor(0.0), out.new_tensor(1.0)

    @classmethod
    def _run_ffn_shard_with_overlap(
        cls,
        module: nn.Module,
        x: torch.Tensor,
        compute_aux_loss: bool,
        token_start: int,
        total_tokens: int,
        token_parallel: TokenParallelRuntime,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not token_parallel.should_overlap(x):
            local_out, aux_loss, sparsity = cls._run_ffn_shard(
                module,
                x,
                compute_aux_loss=compute_aux_loss,
                token_start=token_start,
            )
            gathered, _, _ = token_parallel.all_gather_dim(local_out, dim=1, total_size=total_tokens)
            return gathered, aux_loss, sparsity

        if isinstance(module, SparsePerTokenMoE):
            local_out, router_logits, sparsity = module.forward_shard_deferred_aux(x, token_start)
        else:
            if hasattr(module, "forward_shard"):
                local_out = module.forward_shard(x, token_start)
            else:
                local_out = module(x)
            router_logits = None
            sparsity = local_out.new_tensor(1.0)

        pending_gather = token_parallel.begin_all_gather_dim(local_out, dim=1, total_size=total_tokens)
        if compute_aux_loss and isinstance(module, SparsePerTokenMoE):
            aux_loss = module.compute_aux_loss_from_logits(router_logits, x)
        else:
            aux_loss = x.float().new_tensor(0.0)
        return pending_gather.wait(), aux_loss, sparsity

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.validate_static_shapes:
            if x.dim() != 3:
                raise ValueError(f"TokenMixerLargeBlock expects [B, T, D], got dim={x.dim()}")
            if x.size(1) != self.num_tokens:
                raise ValueError(
                    f"token count changed; keep T static for NPU graph mode, "
                    f"expected {self.num_tokens}, got {x.size(1)}"
                )
            if x.size(2) != self.dim:
                raise ValueError(
                    f"token dim changed; keep D static for NPU graph mode, "
                    f"expected {self.dim}, got {x.size(2)}"
        residual = x
        # Pre-norm flow requested by the TokenMixer-Large block design:
        # Mix -> Norm -> S-P MoE -> dropout -> residual -> Revert ->
        # Norm -> S-P MoE -> dropout -> original residual.
        mixed = x
        if self.use_mixing_reverting:
            mixed = self.mix(mixed)
        mixed_normed = self.mixed_norm(mixed)
        mixed_delta, mixed_aux_loss, mixed_sparsity = self._run_ffn(
            self.mixed_ffn, mixed_normed, compute_aux_loss=self.use_aux_loss
        )
        mixed_next = mixed + self.dropout(mixed_delta)
        if self.use_mixing_reverting:
            original_input = self.revert(mixed_next)
        else:
            original_input = mixed_next
        original_normed = self.original_norm(original_input)
        original_delta, original_aux_loss, original_sparsity = self._run_ffn(
            self.original_ffn, original_normed, compute_aux_loss=self.use_aux_loss
        )
        # Equation (16) keeps the original pre-mixing semantic residual.
        x = residual + self.dropout(original_delta)

        if self.use_aux_loss:
            aux_loss = mixed_aux_loss + original_aux_loss
            if self.activation_aux_scale > 0.0:
                aux_loss = aux_loss + self.activation_aux_scale * (
                    mixed_delta.float().pow(2).mean() + original_delta.float().pow(2).mean()
                )
        else:
            aux_loss = x.float().new_tensor(0.0)
        sparsity = 0.5 * (mixed_sparsity + original_sparsity)
        return x, aux_loss, sparsity

    def forward_token_parallel(
        self,
        x: torch.Tensor,
        token_parallel: TokenParallelRuntime,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.validate_static_shapes:
            if x.dim() != 3:
                raise ValueError(f"TokenMixerLargeBlock expects [B, T, D], got dim={x.dim()}")
            if x.size(1) != self.num_tokens:
                raise ValueError(
                    f"token count changed; keep T static for NPU graph mode, "
                    f"expected {self.num_tokens}, got {x.size(1)}"
                )
            if x.size(2) != self.dim:
                raise ValueError(
                    f"token dim changed; keep D static for NPU graph mode, "
                    f"expected {self.dim}, got {x.size(2)}"
                )
        residual = x
        # Keep the same pre-norm/residual ordering as ``forward`` while
        # sharding only the expensive per-token FFN/MoE computation.
        mixed = x
        if self.use_mixing_reverting:
            mixed = self.mix(mixed)
        mixed_normed = self.mixed_norm(mixed)

        mixed_start, mixed_end = token_parallel.shard_range(mixed_normed.size(1))
        mixed_local = mixed_normed[:, mixed_start:mixed_end, :]
        mixed_delta, mixed_aux_loss, mixed_sparsity = self._run_ffn_shard_with_overlap(
            self.mixed_ffn,
            mixed_local,
            compute_aux_loss=self.use_aux_loss,
            token_start=mixed_start,
            total_tokens=mixed_normed.size(1),
            token_parallel=token_parallel,
        )
        mixed_next = mixed + self.dropout(mixed_delta)
        if self.use_mixing_reverting:
            original_input = self.revert(mixed_next)
        else:
            original_input = mixed_next
        original_normed = self.original_norm(original_input)
        original_start, original_end = token_parallel.shard_range(original_normed.size(1))
        original_local = original_normed[:, original_start:original_end, :]
        original_delta, original_aux_loss, original_sparsity = self._run_ffn_shard_with_overlap(
            self.original_ffn,
            original_local,
            compute_aux_loss=self.use_aux_loss,
            token_start=original_start,
            total_tokens=original_normed.size(1),
            token_parallel=token_parallel,
        )
        x = residual + self.dropout(original_delta)

        if self.use_aux_loss:
            if token_parallel.reduce_aux_loss:
                aux_pair = token_parallel.all_reduce_mean(torch.stack([mixed_aux_loss, original_aux_loss]))
                mixed_aux_loss = aux_pair[0]
                original_aux_loss = aux_pair[1]
            aux_loss = mixed_aux_loss + original_aux_loss
            if self.activation_aux_scale > 0.0:
                aux_loss = aux_loss + self.activation_aux_scale * (
                    mixed_delta.float().pow(2).mean() + original_delta.float().pow(2).mean()
                )
        else:
            aux_loss = x.float().new_tensor(0.0)
        sparsity = 0.5 * (mixed_sparsity + original_sparsity)
        return x, aux_loss, sparsity


@ModelRegistry.register()
class TokenMixerLarge(BaseModel):
    def __init__(self, model_cfg: Dict, common_hp: Dict, model_cls_dict: Dict) -> None:
        super().__init__(model_cfg=model_cfg, common_hp=common_hp, model_cls_dict=model_cls_dict)
        model_conf = common_hp["model_conf"]
        hp = model_cfg[Const.HP]

        self._embedding_dim: int = model_conf.get("item_embedding_dim", 256)
        self.num_base_tokens = hp.get("num_tokens", hp.get("T", 0))
        self.dim_per_token = hp.get("dim_per_token", self._embedding_dim)
        self.use_global_token = hp.get("use_global_token", True)
        self.num_tokens = self.num_base_tokens + (1 if self.use_global_token else 0)
        self.num_heads = hp.get("num_heads", max(1, min(8, self.num_tokens)))
        self.hidden_multiplier = hp.get("swiglu_multiplier", hp.get("k", 4))
        self.dropout_p = hp.get("dropout", 0.05)
        self.bias = hp.get("bias", True)
        self.down_init_scale = hp.get("down_init_scale", 0.01)
        self.n_layers = hp.get("n_layers", 2)
        self.norm_eps = hp.get("norm_eps", Const.EPS)

        self.use_rmsnorm = hp.get("use_rmsnorm", True)
        configured_fused_add_rmsnorm = hp.get("use_fused_add_rmsnorm", True)
        fused_add_rmsnorm_env = os.environ.get("USE_FUSED_ADD_RMSNORM")
        self.use_fused_add_rmsnorm = (
            configured_fused_add_rmsnorm
            if fused_add_rmsnorm_env is None
            else fused_add_rmsnorm_env != "0"
        )
        self.use_pswiglu = hp.get("use_pswiglu", True)
        self.use_mixing_reverting = hp.get("use_mixing_reverting", True)
        self.aux_loss_weight = hp.get("aux_loss_weight", 0.0)
        self.use_aux_loss = hp.get("use_aux_loss", self.aux_loss_weight > 0.0)
        self.activation_aux_scale = hp.get("activation_aux_scale", 1e-4)
        self.use_inter_residual = hp.get("use_inter_residual", hp.get("interval_residual", 0) > 0)
        interval = hp.get("inter_residual_interval", hp.get("interval_residual", 2))
        self.inter_residual_interval = 2 if self.use_inter_residual and interval <= 0 else interval

        sparse_moe = hp.get("sparseMOE", False)
        configured_moe_type = str(hp.get("moe_type", "dense")).lower()
        self.moe_type = "spmoe" if sparse_moe else configured_moe_type
        self.num_experts = hp.get("num_experts", hp.get("num_experts_per_token", 4))
        self.moe_topk = hp.get("moe_topk", hp.get("top_k_number", 2))
        self.moe_shared_expert = hp.get("moe_shared_expert", hp.get("use_shared_expert", True))
        self.gate_scale_alpha = hp.get("gate_scale_alpha", hp.get("gate_scale"))
        self.npu_bmm = hp.get("npu_bmm", hp.get("use_npu_bmm", True))
        self.fused_gate_up = hp.get("fused_gate_up", True)
        self.use_npu_swiglu = hp.get("use_npu_swiglu", True)
        self.sparse_top1_dispatch = hp.get("sparse_top1_dispatch", True)
        self.fast_top1_gating = hp.get("fast_top1_gating", True)
        self.use_npu_grouped_moe = hp.get("use_npu_grouped_moe", True)
        configured_grouped_bias_fusion = hp.get("use_npu_grouped_moe_bias_fusion", True)
        grouped_bias_fusion_env = os.environ.get("NPU_GROUPED_MOE_BIAS_FUSION")
        self.use_npu_grouped_moe_bias_fusion = (
            configured_grouped_bias_fusion
            if grouped_bias_fusion_env is None
            else grouped_bias_fusion_env != "0"
        )
        self.fallback_on_npu_grouped_moe_failure = hp.get("fallback_on_npu_grouped_moe_failure", True)
        self.validate_static_shapes = hp.get("validate_static_shapes", True)
        self.use_token_parallel = hp.get("use_token_parallel", False)
        self.token_parallel_size = hp.get("token_parallel_size", 0)
        self.tp_reduce_aux_loss = hp.get("tp_reduce_aux_loss", False)
        configured_overlap = hp.get("tp_comm_overlap", True)
        overlap_env = os.environ.get("TP_COMM_OVERLAP")
        self.tp_comm_overlap = configured_overlap if overlap_env is None else overlap_env != "0"
        configured_comm_dtype = hp.get("tp_comm_dtype", "float32")
        self.tp_comm_dtype = os.environ.get("TP_COMM_DTYPE", configured_comm_dtype)
        self.token_parallel = TokenParallelRuntime(
            self.use_token_parallel,
            self.token_parallel_size,
            reduce_aux_loss=self.tp_reduce_aux_loss,
            comm_overlap=self.tp_comm_overlap,
            comm_dtype=self.tp_comm_dtype,
        )

        self.input_proj = nn.Linear(self._embedding_dim, self.dim_per_token, bias=self.bias)
        if self.use_global_token:
            self.global_proj = nn.Sequential(
                nn.Linear(self.dim_per_token, self.dim_per_token, bias=self.bias),
                nn.SiLU(),
                nn.Linear(self.dim_per_token, self.dim_per_token, bias=self.bias),
            )
        else:
            self.global_proj = None

        self.blocks = nn.ModuleList(
            [
                TokenMixerLargeBlock(
                    num_tokens=self.num_tokens,
                    dim=self.dim_per_token,
                    num_heads=self.num_heads,
                    hidden_multiplier=self.hidden_multiplier,
                    dropout_p=self.dropout_p,
                    bias=self.bias,
                    down_init_scale=self.down_init_scale,
                    use_rmsnorm=self.use_rmsnorm,
                    use_pswiglu=self.use_pswiglu,
                    use_mixing_reverting=self.use_mixing_reverting,
                    use_aux_loss=self.use_aux_loss,
                    activation_aux_scale=self.activation_aux_scale,
                    moe_type=self.moe_type,
                    num_experts=self.num_experts,
                    moe_topk=self.moe_topk,
                    gate_scale_alpha=self.gate_scale_alpha,
                    moe_shared_expert=self.moe_shared_expert,
                    norm_eps=self.norm_eps,
                    npu_bmm=self.npu_bmm,
                    fused_gate_up=self.fused_gate_up,
                    use_npu_swiglu=self.use_npu_swiglu,
                    sparse_top1_dispatch=self.sparse_top1_dispatch,
                    fast_top1_gating=self.fast_top1_gating,
                    use_npu_grouped_moe=self.use_npu_grouped_moe,
                    use_npu_grouped_moe_bias_fusion=self.use_npu_grouped_moe_bias_fusion,
                    fallback_on_npu_grouped_moe_failure=self.fallback_on_npu_grouped_moe_failure,
                    use_fused_add_rmsnorm=self.use_fused_add_rmsnorm,
                    validate_static_shapes=self.validate_static_shapes,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.output_proj = nn.Linear(self.dim_per_token, self._embedding_dim, bias=False)
        self.balance_loss_coef = 1e-8

    def _make_tokens(self, item_feature_embs: torch.Tensor) -> torch.Tensor:
        tokens = self.input_proj(item_feature_embs)
        if self.num_base_tokens > 0:
            tokens = tokens[:, : self.num_base_tokens, :]
        if self.global_proj is not None:
            global_token = self.global_proj(tokens.mean(dim=1, keepdim=True))
            tokens = torch.cat([global_token, tokens], dim=1)
        return tokens

    def forward(self, past_ids, num_rerank, model_inputs, user_feature_embs, item_feature_embs):
        batch_size, num_items, _ = item_feature_embs.size()
        local_batch_size = batch_size
        local_batch_start = 0
        x = self._make_tokens(item_feature_embs)
        if self.token_parallel.enabled:
            x, _, local_batch_start = self.token_parallel.all_gather_dim(
                x,
                dim=0,
                total_size=batch_size * self.token_parallel.group_size,
            )
            batch_size = x.size(0)
        aux_loss = x.float().new_tensor(0.0)
        sparsity = x.new_tensor(0.0)
        interval_anchor = x
        for layer_idx, block in enumerate(self.blocks):
            if self.token_parallel.enabled:
                x, layer_aux_loss, layer_sparsity = block.forward_token_parallel(
                    x,
                    self.token_parallel,
                )
            else:
                x, layer_aux_loss, layer_sparsity = block(x)
            aux_loss = aux_loss + layer_aux_loss
            sparsity = sparsity + layer_sparsity
            if (
                self.use_inter_residual
                and self.inter_residual_interval
                and (layer_idx + 1) % self.inter_residual_interval == 0
                and layer_idx + 1 < len(self.blocks)
            ):
                x = x + interval_anchor
                interval_anchor = x

        pooled = x.mean(dim=1)
        y = self.output_proj(pooled).view(batch_size, 1, -1).expand(-1, num_items, -1)
        if self.token_parallel.enabled:
            y = y.narrow(0, local_batch_start, local_batch_size)
        denom = max(1, len(self.blocks))
        aux_loss = aux_loss / denom
        weighted_aux_loss = aux_loss * self.aux_loss_weight if self.use_aux_loss else aux_loss.new_zeros(())
        return {
            "deep_outputs": y,
            "deep_loss": aux_loss,
            "deep_sparsity": sparsity / denom,
            "aux_loss": weighted_aux_loss,
        }
