# pylint: disable=too-many-lines
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_common_utils import allclose, jagged_to_dense, dense_to_jagged, MaskType, QKVShapeInfo, MaskGenInfo
from test_target_mask import create_causal_mask


def jagged_data_gen(
    qkv_shape_info: QKVShapeInfo, mask_info: MaskGenInfo, enable_bias: bool, repeat_offset: bool = False
):
    int_type = qkv_shape_info.int_type
    batch_size = qkv_shape_info.batch_size
    max_num_context, max_num_target = mask_info.max_num_context, mask_info.max_num_target
    min_num_context, min_num_target = mask_info.max_num_context, mask_info.max_num_target
    num_context = torch.randint(min_num_context, max_num_context + 1, (batch_size,), dtype=int_type)
    num_target = torch.randint(min_num_target, max_num_target + 1, (batch_size,), dtype=int_type)

    float_type = qkv_shape_info.float_type
    min_seq_len, max_seq_len = qkv_shape_info.min_seq_len, qkv_shape_info.max_seq_len

    seq_lens = torch.randint(min_seq_len, max_seq_len + 1, (batch_size,))
    if mask_info.mask_type == MaskType.TRIL:
        seq_lens += num_context + num_target
    seq_offset = torch.concat((torch.zeros((1,)), torch.cumsum(seq_lens, dim=0))).to(int_type)
    if repeat_offset:
        seq_offset = torch.cat((seq_offset, seq_offset[-1:]), dim=0)
        num_context = torch.cat((num_context, num_context[-1:]), dim=0)
        num_target = torch.cat((num_target, num_target[-1:]), dim=0)
    max_seq_len, total_seqs = max(seq_lens.tolist()), sum(seq_lens.tolist())

    num_heads_q, num_heads_k, head_dim_qk, head_dim_v = (
        qkv_shape_info.num_heads_q,
        qkv_shape_info.num_heads_k,
        qkv_shape_info.head_dim_qk,
        qkv_shape_info.head_dim_v,
    )

    if float_type == torch.float8_e4m3fn:
        data_type = torch.float16
    else:
        data_type = float_type

    q = torch.rand(total_seqs, num_heads_q, head_dim_qk, dtype=data_type).uniform_(-1, 1)
    k = torch.rand(total_seqs, num_heads_k, head_dim_qk, dtype=data_type).uniform_(-1, 1)
    v = torch.rand(total_seqs, num_heads_k, head_dim_v, dtype=data_type).uniform_(-1, 1)

    rel_attn_bias = (
        torch.rand(batch_size, num_heads_q, max_seq_len, max_seq_len).to(float_type) if enable_bias else None
    )

    if mask_info.mask_type == MaskType.TRIL:
        mask = torch.zeros(batch_size, num_heads_q, max_seq_len, max_seq_len)
        seqs = zip(seq_lens.tolist(), num_context.tolist(), num_target.tolist())
        for batch_id, (seq_len, ctx, tar) in enumerate(seqs):
            mask_tensor = create_causal_mask(seq_len, seq_len, ctx, tar, mask_info.target_group_size)
            mask[batch_id, :, :seq_len, :seq_len] = mask_tensor
        mask = mask.cpu().to(float_type)
    elif mask_info.mask_type == MaskType.CUSTOM:
        mask = torch.randint(0, 2, size=(batch_size, num_heads_q, max_seq_len, max_seq_len))
        mask = mask.cpu().to(float_type)
    else:
        mask = None

    if float_type == torch.float8_e4m3fn:
        q = q.to(torch.float8_e4m3fn)
        k = k.to(torch.float8_e4m3fn)
        v = v.to(torch.float8_e4m3fn)

    qkv_tensors = (q, k, v, seq_offset)
    mask_tensors = (mask_info.mask_type, mask, num_context, num_target, mask_info.target_group_size)
    return qkv_tensors, mask_tensors, rel_attn_bias, max_seq_len


class TestHstuJaggedDemo:
    @staticmethod
    def custom_op_exec(qkv_tensors, mask_tensors, rel_attn_bias, silu_scale, max_seq_len, deterministic=False):
        q, k, v, seq_offset = qkv_tensors
        mask_type, mask, num_context, num_target, target_group_size = mask_tensors

        q_npu = q.to("npu")
        k_npu = k.to("npu")
        v_npu = v.to("npu")
        bias_npu = rel_attn_bias.to("npu") if isinstance(rel_attn_bias, torch.Tensor) else None
        mask_npu = mask.to("npu") if mask_type is MaskType.CUSTOM else None
        seq_offset = seq_offset.to("npu")
        num_context = num_context.to("npu")
        num_target = num_target.to("npu")
        # 函数重载：hstu_jagged -> hstu_jagged.equal
        output = torch.ops.mxrec.hstu_jagged(
            q_npu,
            k_npu,
            v_npu,
            mask_npu,
            bias_npu,
            mask_type,
            max_seq_len,
            silu_scale,
            seq_offset,
            num_context,
            num_target,
            target_group_size,
            deterministic=deterministic,
        )
        torch.npu.synchronize()
        return output.cpu()

    @staticmethod
    def golden_op_exec(qkv_tensors, mask_tensors, rel_attn_bias, silu_scale, max_seq_len, repeat_offset):
        q, k, v, seq_offset = qkv_tensors
        mask_type, mask, _, _, _ = mask_tensors

        (_, head_nums_q, head_dim), data_type = q.shape, q.dtype
        (_, head_nums_k, head_dim) = k.shape
        head_dim_v = v.shape[2]
        batch_size = seq_offset.shape[0] - 1 - int(repeat_offset)

        if head_nums_q != head_nums_k:
            assert head_nums_q % head_nums_k == 0, (
                f"head_nums_q ({head_nums_q}) must be divisible by head_nums_k({head_nums_k}) "
            )
        h_qk_ratio = head_nums_q // head_nums_k

        seq_lens = np.zeros((batch_size,)).astype(np.int32)
        for batch_id in range(batch_size):
            seq_lens[batch_id] = seq_offset[batch_id + 1] - seq_offset[batch_id]

        silu_scale = 1 / max_seq_len if silu_scale == 0 else silu_scale

        use_float8 = data_type == torch.float8_e4m3fn
        if use_float8:
            q_dens = jagged_to_dense(q, seq_lens, head_nums_q, head_dim).to(torch.float32).to("npu")
            k_dens = jagged_to_dense(k, seq_lens, head_nums_k, head_dim).to(torch.float32).to("npu")
            v_dens = jagged_to_dense(v, seq_lens, head_nums_k, head_dim_v).to(torch.float32).to("npu")
        else:
            q_dens = jagged_to_dense(q, seq_lens, head_nums_q, head_dim).to(data_type).to("npu")
            k_dens = jagged_to_dense(k, seq_lens, head_nums_k, head_dim).to(data_type).to("npu")
            v_dens = jagged_to_dense(v, seq_lens, head_nums_k, head_dim_v).to(data_type).to("npu")

        k_dens_expanded = k_dens.repeat_interleave(h_qk_ratio, dim=2)
        v_dens_expanded = v_dens.repeat_interleave(h_qk_ratio, dim=2)

        q_dens = q_dens.permute(0, 2, 1, 3)
        k_dens = k_dens_expanded.permute(0, 2, 3, 1)
        qk_attn = torch.matmul(q_dens, k_dens).to(torch.float32)

        if rel_attn_bias is not None:
            rel_attn_bias = rel_attn_bias.to(torch.float32).to("npu")
            qk_attn += rel_attn_bias

        F.silu(qk_attn, inplace=True)
        qk_attn *= silu_scale

        if mask_type != MaskType.NONE:
            mask = mask.to(torch.float32).to("npu")
            qk_attn *= mask

        v_dens = v_dens_expanded.permute(0, 2, 1, 3)

        if use_float8:
            qk_attn = qk_attn.to(torch.float8_e4m3fn)
            qk_attn = qk_attn.to(torch.float32)
        else:
            qk_attn = qk_attn.to(data_type)

        attn_output = torch.matmul(qk_attn, v_dens)
        attn_output = attn_output.permute(0, 2, 1, 3).cpu()
        attn_output = dense_to_jagged(q, attn_output, seq_lens)

        if use_float8:
            attn_output = attn_output.to(torch.float16)

        torch.npu.synchronize()
        return attn_output.to(torch.float16) if use_float8 else attn_output.to(data_type)

    def execute(self, qkv_shape_info, mask_info, enable_bias, silu_scale, repeat_offset=False, deterministic=False):
        qkv_tensors, mask_tensors, rel_attn_bias, max_seq_len = jagged_data_gen(
            qkv_shape_info, mask_info, enable_bias, repeat_offset
        )

        output = self.custom_op_exec(qkv_tensors, mask_tensors, rel_attn_bias, silu_scale, max_seq_len, deterministic)
        golden = self.golden_op_exec(qkv_tensors, mask_tensors, rel_attn_bias, silu_scale, max_seq_len, repeat_offset)

        data_type = qkv_shape_info.float_type
        if data_type == torch.bfloat16:
            res = allclose(output, golden, 5e-3, 5e-3)
        elif data_type == torch.float8_e4m3fn:
            res = allclose(output, golden, 1e-3, 1e-3)
        elif data_type == torch.float16:
            res = allclose(output, golden, 1e-3, 1e-3)
        else:
            res = allclose(output, golden, 1e-4, 1e-4)
        assert res

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("max_seq_len", [15, 1024])
    @pytest.mark.parametrize("head_dim", [16, 128])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int32])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 1, 0, 30),
            (MaskType.TRIL, 3, 0, 30),
            (MaskType.TRIL, 1, 6, 0),
            (MaskType.TRIL, 3, 6, 0),
            (MaskType.TRIL, 1, 6, 30),
            (MaskType.TRIL, 3, 6, 30),
        ],
    )
    def test_hstu_jagged_forward(
        self,
        batch_size,
        head_num,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize(
        "target_group_size, max_num_context, max_num_target",
        [(0, 0, 0), (1, 255, 30), (1, 256, 30), (1, 257, 30), (3, 20, 511), (3, 20, 512)],
    )
    def test_hstu_jagged_forward_mask(self, target_group_size, max_num_context, max_num_target):
        qkv_shape_info = QKVShapeInfo(
            float_type=torch.float16,
            int_type=torch.int32,
            batch_size=1,
            max_seq_len=16,
            num_heads_q=1,
            num_heads_k=1,
            head_dim_qk=16,
            head_dim_v=16,
        )
        mask_info = MaskGenInfo(
            mask_type=MaskType.TRIL,
            max_num_context=max_num_context,
            min_num_context=max_num_context,
            max_num_target=max_num_target,
            min_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, False, 0)

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("head_num", [1, 7, 16])
    @pytest.mark.parametrize("max_seq_len", [15, 256])
    @pytest.mark.parametrize("head_dim", [16, 32])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("mask_type", [MaskType.NONE, MaskType.CUSTOM, MaskType.TRIL])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int32])
    @pytest.mark.parametrize("repeat_offset", [False, True])
    def test_hstu_jagged_forward_head16(
        self,
        batch_size,
        head_num,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        repeat_offset,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(mask_type=mask_type, max_num_context=0, max_num_target=0, target_group_size=0)
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale, repeat_offset)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [2570])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int32])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 1, 6, 30),
        ],
    )
    def test_hstu_jagged_forward_128bs(
        self,
        head_num,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=128,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("head_num", [2])
    @pytest.mark.parametrize("max_seq_len", [16])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 3, 6, 30),
        ],
    )
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_jagged_forward_2048bs(
        self,
        head_num,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=torch.int32,
            batch_size=2048,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    def test_error_nhead_255(self):
        qkv_shape_info = QKVShapeInfo(
            float_type=torch.float16,
            int_type=torch.int32,
            batch_size=20,
            max_seq_len=16,
            num_heads_q=255,
            num_heads_k=255,
            head_dim_qk=256,
            head_dim_v=256,
        )
        mask_info = MaskGenInfo(mask_type=MaskType.NONE)
        with pytest.raises(RuntimeError) as e_info:
            self.execute(qkv_shape_info, mask_info, False, 0)
            assert "head num must meet range[1 16] and mutiple of [1]. but get value 255" in str(e_info.value)

    def test_error_head_dim_255(self):
        qkv_shape_info = QKVShapeInfo(
            float_type=torch.float16,
            int_type=torch.int32,
            batch_size=20,
            max_seq_len=16,
            num_heads_q=2,
            num_heads_k=2,
            head_dim_qk=255,
            head_dim_v=255,
        )
        mask_info = MaskGenInfo(mask_type=MaskType.NONE)
        with pytest.raises(RuntimeError) as e_info:
            self.execute(qkv_shape_info, mask_info, False, 0)
            assert "dim size must meet range[16 512] and mutiple of [16]. but get value 255" in str(e_info.value)

    # ============ 异常场景测试用例 ============

    @staticmethod
    def _valid_shape_info(**overrides):
        """构造一个合法的 QKVShapeInfo 基线，可按需覆盖字段"""
        params = dict(
            float_type=torch.float16,
            int_type=torch.int32,
            batch_size=2,
            max_seq_len=16,
            num_heads_q=2,
            num_heads_k=2,
            head_dim_qk=64,
            head_dim_v=64,
        )
        params.update(overrides)
        return QKVShapeInfo(**params)

    def _make_valid_npu_inputs(self):
        """生成一组合法的 NPU 输入并搬到 device，供直接调用算子触发 host 侧校验"""
        qkv_shape_info = self._valid_shape_info()
        mask_info = MaskGenInfo(mask_type=MaskType.NONE)
        qkv_tensors, mask_tensors, _, max_seq_len = jagged_data_gen(qkv_shape_info, mask_info, enable_bias=False)
        q, k, v, seq_offset = qkv_tensors
        _, _, num_context, num_target, target_group_size = mask_tensors
        return {
            "q": q.to("npu"),
            "k": k.to("npu"),
            "v": v.to("npu"),
            "seq_offset": seq_offset.to("npu"),
            "num_context": num_context.to("npu"),
            "num_target": num_target.to("npu"),
            "target_group_size": target_group_size,
            "max_seq_len": max_seq_len,
        }

    @staticmethod
    def _call_op(t, q=None, mask=None, mask_type=MaskType.NONE, max_seq_len=None):
        """直接调用 hstu_jagged.equal，参数可覆盖以构造异常场景"""
        output = torch.ops.mxrec.hstu_jagged(
            q if q is not None else t["q"],
            t["k"],
            t["v"],
            mask,
            None,
            mask_type,
            max_seq_len if max_seq_len is not None else t["max_seq_len"],
            0.0,
            t["seq_offset"],
            t["num_context"],
            t["num_target"],
            t["target_group_size"],
        )
        torch.npu.synchronize()
        return output

    def test_jagged_q_not_3d(self):
        """jagged layout 要求 q 是 3D，传入非 3D 应报错"""
        t = self._make_valid_npu_inputs()
        q_bad = t["q"].reshape(t["q"].shape[0], -1)  # 3D -> 2D
        with pytest.raises(Exception) as ctx:
            self._call_op(t, q=q_bad)
        assert "The q should be 3D in jagged layout" in str(ctx.value)

    def test_jagged_max_seq_len_too_large(self):
        """max_seq_len > 20480，触发 MaxSeqLenCheck"""
        t = self._make_valid_npu_inputs()
        with pytest.raises(Exception) as ctx:
            self._call_op(t, max_seq_len=20481)
        assert "maxSeqLen expect in [1, 20480]" in str(ctx.value)

    def test_jagged_max_seq_len_too_small(self):
        """max_seq_len < 1，触发 MaxSeqLenCheck 下界"""
        t = self._make_valid_npu_inputs()
        with pytest.raises(Exception) as ctx:
            self._call_op(t, max_seq_len=0)
        assert "maxSeqLen expect in [1, 20480]" in str(ctx.value)

    def test_jagged_seq_offset_too_short(self):
        """seq_offset 元素数 < 2，触发 acSeqOffset 校验"""
        t = self._make_valid_npu_inputs()
        t["seq_offset"] = torch.tensor([0], dtype=torch.int32, device="npu")
        with pytest.raises(Exception) as ctx:
            self._call_op(t)
        assert "acSeqOffset params error should have at least two element" in str(ctx.value)

    def test_jagged_mask_type_triu_not_supported(self):
        """mask_type=TRIU(1) 不支持，触发 MaskCheck"""
        t = self._make_valid_npu_inputs()
        with pytest.raises(Exception) as ctx:
            self._call_op(t, mask_type=MaskType.TRIU)
        assert "maskType current not support triu now" in str(ctx.value)

    def test_jagged_custom_mask_missing(self):
        """mask_type=CUSTOM(3) 但未提供 mask tensor，触发 MaskCheck"""
        t = self._make_valid_npu_inputs()
        with pytest.raises(Exception) as ctx:
            self._call_op(t, mask=None, mask_type=MaskType.CUSTOM)
        assert "use custom mask must have valid mask tensor" in str(ctx.value)

    def test_jagged_mask_type_out_of_range(self):
        """mask_type 超出 [0, 3]，触发 MaskCheck"""
        t = self._make_valid_npu_inputs()
        with pytest.raises(Exception) as ctx:
            self._call_op(t, mask_type=4)
        assert "maskType expect in [0, 3]" in str(ctx.value)

    def test_jagged_batch_size_exceeds_2048(self):
        """batch_size > 2048 triggers kernel validation"""
        qkv_shape_info = self._valid_shape_info(batch_size=2049, max_seq_len=512)
        mask_info = MaskGenInfo(mask_type=MaskType.NONE)
        with pytest.raises((RuntimeError, ValueError)):
            self.execute(qkv_shape_info, mask_info, enable_bias=False, silu_scale=0.0)

    @pytest.mark.parametrize("batch_size", [2560, 3000, 4096])
    def test_jagged_batch_size_multiple_exceeds_2048(self, batch_size):
        """batch_size values > 2048 trigger kernel validation"""
        qkv_shape_info = self._valid_shape_info(batch_size=batch_size, max_seq_len=256)
        mask_info = MaskGenInfo(mask_type=MaskType.NONE)
        with pytest.raises((RuntimeError, ValueError)):
            self.execute(qkv_shape_info, mask_info, enable_bias=False, silu_scale=0.0)

    def test_jagged_num_context_shape_mismatch(self):
        """num_context shape mismatch with batch_size triggers kernel validation"""
        qkv_shape_info = self._valid_shape_info(batch_size=32, max_seq_len=128)
        mask_info = MaskGenInfo(mask_type=MaskType.TRIL, max_num_context=10, target_group_size=2)
        qkv_tensors, mask_tensors, _, max_seq_len = jagged_data_gen(qkv_shape_info, mask_info, enable_bias=False)
        q, k, v, seq_offset = qkv_tensors
        mask, _, _, num_target, target_group_size = mask_tensors
        # Create wrong-sized num_context
        num_context_wrong = torch.ones(16, device="npu", dtype=qkv_shape_info.int_type) * 10
        with pytest.raises((RuntimeError, ValueError)):
            self.custom_op_exec(
                (q, k, v, seq_offset),
                (mask, None, num_context_wrong, num_target, target_group_size),
                None,
                0.0,
                max_seq_len,
            )

    def test_jagged_num_target_shape_mismatch(self):
        """num_target shape mismatch with batch_size triggers kernel validation"""
        qkv_shape_info = self._valid_shape_info(batch_size=32, max_seq_len=128)
        mask_info = MaskGenInfo(mask_type=MaskType.TRIL, max_num_target=20, target_group_size=2)
        qkv_tensors, mask_tensors, _, max_seq_len = jagged_data_gen(qkv_shape_info, mask_info, enable_bias=False)
        q, k, v, seq_offset = qkv_tensors
        mask, _, num_context, _, target_group_size = mask_tensors
        # Create wrong-sized num_target
        num_target_wrong = torch.ones(64, device="npu", dtype=qkv_shape_info.int_type) * 20
        with pytest.raises((RuntimeError, ValueError)):
            self.custom_op_exec(
                (q, k, v, seq_offset),
                (mask, None, num_context, num_target_wrong, target_group_size),
                None,
                0.0,
                max_seq_len,
            )

    def test_jagged_seqlens_k_not_equal_seqlens_v(self):
        """seqlensK != seqlensV triggers kernel validation"""
        qkv_shape_info = self._valid_shape_info(batch_size=16, max_seq_len=128)
        mask_info = MaskGenInfo(mask_type=MaskType.NONE)
        qkv_tensors, mask_tensors, _, max_seq_len = jagged_data_gen(qkv_shape_info, mask_info, enable_bias=False)
        q, k, v, seq_offset = qkv_tensors
        mask, _, num_context, num_target, target_group_size = mask_tensors
        # Create V with wrong length
        total_len_k = seq_offset[-1].item()
        v_wrong = torch.rand(
            total_len_k + 100,
            qkv_shape_info.num_heads_k,
            qkv_shape_info.head_dim_v,
            device="npu",
            dtype=qkv_shape_info.float_type,
        )
        with pytest.raises((RuntimeError, ValueError)):
            self.custom_op_exec(
                (q, k, v_wrong, seq_offset),
                (mask, None, num_context, num_target, target_group_size),
                None,
                0.0,
                max_seq_len,
            )

    def test_jagged_head_num_k_not_equal_head_num_v(self):
        """headNumK != headNumV triggers kernel validation"""
        qkv_shape_info = self._valid_shape_info(batch_size=16, num_heads_k=4, max_seq_len=128)
        mask_info = MaskGenInfo(mask_type=MaskType.NONE)
        qkv_tensors, mask_tensors, _, max_seq_len = jagged_data_gen(qkv_shape_info, mask_info, enable_bias=False)
        q, k, v, seq_offset = qkv_tensors
        mask, _, num_context, num_target, target_group_size = mask_tensors
        # Create V with wrong head count
        total_len_k = seq_offset[-1].item()
        v_wrong = torch.rand(total_len_k, 8, qkv_shape_info.head_dim_v, device="npu", dtype=qkv_shape_info.float_type)
        with pytest.raises((RuntimeError, ValueError)):
            self.custom_op_exec(
                (q, k, v_wrong, seq_offset),
                (mask, None, num_context, num_target, target_group_size),
                None,
                0.0,
                max_seq_len,
            )

    @pytest.mark.parametrize("num_heads", [17, 20, 32])
    def test_jagged_head_num_out_of_range(self, num_heads):
        """headNum > 16 triggers kernel validation (num_heads < 1 generates empty tensors, no error)"""
        qkv_shape_info = self._valid_shape_info(
            batch_size=8, num_heads_q=num_heads, num_heads_k=num_heads, max_seq_len=128
        )
        mask_info = MaskGenInfo(mask_type=MaskType.NONE)
        with pytest.raises((RuntimeError, ValueError)):
            self.execute(qkv_shape_info, mask_info, enable_bias=False, silu_scale=0.0)

    @pytest.mark.parametrize("head_dim", [520, 600, 1024])
    def test_jagged_head_dim_exceeds_512(self, head_dim):
        """headDim > 512 triggers kernel validation"""
        qkv_shape_info = self._valid_shape_info(
            batch_size=8, head_dim_qk=head_dim, head_dim_v=head_dim, max_seq_len=128
        )
        mask_info = MaskGenInfo(mask_type=MaskType.NONE)
        with pytest.raises((RuntimeError, ValueError)):
            self.execute(qkv_shape_info, mask_info, enable_bias=False, silu_scale=0.0)

    @pytest.mark.parametrize("head_dim_v", [17, 33, 100, 127])
    def test_jagged_head_dim_v_not_divisible_by_16(self, head_dim_v):
        """headDimV % 16 != 0 triggers kernel validation"""
        qkv_shape_info = self._valid_shape_info(batch_size=8, head_dim_v=head_dim_v, max_seq_len=128)
        mask_info = MaskGenInfo(mask_type=MaskType.NONE)
        with pytest.raises((RuntimeError, ValueError)):
            self.execute(qkv_shape_info, mask_info, enable_bias=False, silu_scale=0.0)

    @pytest.mark.parametrize("num_heads_q, num_heads_k", [(7, 4), (10, 3), (15, 4)])
    def test_jagged_head_num_q_not_divisible_by_head_num_k(self, num_heads_q, num_heads_k):
        """headNumQ % headNumK != 0 triggers kernel validation"""
        qkv_shape_info = self._valid_shape_info(
            batch_size=8, num_heads_q=num_heads_q, num_heads_k=num_heads_k, max_seq_len=128
        )
        mask_info = MaskGenInfo(mask_type=MaskType.NONE)
        with pytest.raises((RuntimeError, ValueError)):
            self.execute(qkv_shape_info, mask_info, enable_bias=False, silu_scale=0.0)

    def test_jagged_head_dim_q_not_equal_head_dim_k(self):
        """headDimQ != headDimK triggers kernel validation"""
        qkv_shape_info = self._valid_shape_info(batch_size=16, head_dim_qk=128, max_seq_len=128)
        mask_info = MaskGenInfo(mask_type=MaskType.NONE)
        qkv_tensors, mask_tensors, _, max_seq_len = jagged_data_gen(qkv_shape_info, mask_info, enable_bias=False)
        q, k, v, seq_offset = qkv_tensors
        mask, _, num_context, num_target, target_group_size = mask_tensors
        # Create K with wrong head_dim
        total_len_k = seq_offset[-1].item()
        k_wrong = torch.rand(
            total_len_k, qkv_shape_info.num_heads_k, 256, device="npu", dtype=qkv_shape_info.float_type
        )
        with pytest.raises((RuntimeError, ValueError)):
            self.custom_op_exec(
                (q, k_wrong, v, seq_offset),
                (mask, None, num_context, num_target, target_group_size),
                None,
                0.0,
                max_seq_len,
            )

    ## GQA测试
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("head_num_q", [8])
    @pytest.mark.parametrize("head_num_k", [8, 4, 2, 1])
    @pytest.mark.parametrize("max_seq_len", [15, 1024])
    @pytest.mark.parametrize("head_dim", [16, 128])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int32])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 1, 0, 30),
            (MaskType.TRIL, 3, 0, 30),
            (MaskType.TRIL, 1, 6, 0),
            (MaskType.TRIL, 3, 6, 0),
            (MaskType.TRIL, 1, 6, 30),
            (MaskType.TRIL, 3, 6, 30),
        ],
    )
    def test_hstu_jagged_forward_GQA(
        self,
        batch_size,
        head_num_q,
        head_num_k,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads_q=head_num_q,
            num_heads_k=head_num_k,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("head_num_q", [2])
    @pytest.mark.parametrize("head_num_k", [2, 1])
    @pytest.mark.parametrize("max_seq_len", [2570])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int32])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 1, 0, 30),
            (MaskType.TRIL, 3, 0, 30),
            (MaskType.TRIL, 1, 6, 0),
            (MaskType.TRIL, 3, 6, 0),
            (MaskType.TRIL, 1, 6, 30),
            (MaskType.TRIL, 3, 6, 30),
        ],
    )
    def test_hstu_jagged_forward_128bs_GQA(
        self,
        head_num_q,
        head_num_k,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=128,
            max_seq_len=max_seq_len,
            num_heads_q=head_num_q,
            num_heads_k=head_num_k,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("head_num_q", [4])
    @pytest.mark.parametrize("head_num_k", [4, 2, 1])
    @pytest.mark.parametrize("max_seq_len", [16])
    @pytest.mark.parametrize("head_dim", [256])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 3, 6, 30),
        ],
    )
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    def test_hstu_jagged_forward_2048bs_GQA(
        self,
        head_num_q,
        head_num_k,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=torch.int32,
            batch_size=2048,
            max_seq_len=max_seq_len,
            num_heads_q=head_num_q,
            num_heads_k=head_num_k,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    # qk_dim != v_dim
    @pytest.mark.parametrize("batch_size", [4, 128, 1024])
    @pytest.mark.parametrize("head_num_q", [4])
    @pytest.mark.parametrize("head_num_k", [4, 2, 1])
    @pytest.mark.parametrize("max_seq_len", [15])
    @pytest.mark.parametrize("head_dim_qk", [64, 96, 128])
    @pytest.mark.parametrize("head_dim_v", [16, 32, 48, 64, 80])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int32])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 1, 0, 30),
            (MaskType.TRIL, 3, 0, 30),
            (MaskType.TRIL, 1, 6, 0),
            (MaskType.TRIL, 3, 6, 0),
            (MaskType.TRIL, 1, 6, 30),
            (MaskType.TRIL, 3, 6, 30),
        ],
    )
    def test_hstu_jagged_forward_VDA(
        self,
        batch_size,
        head_num_q,
        head_num_k,
        max_seq_len,
        head_dim_qk,
        head_dim_v,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads_q=head_num_q,
            num_heads_k=head_num_k,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)

    @pytest.mark.parametrize("batch_size", [1, 16])
    @pytest.mark.parametrize("head_num", [2, 4])
    @pytest.mark.parametrize("max_seq_len", [15, 1024])
    @pytest.mark.parametrize("head_dim", [16, 128])
    @pytest.mark.parametrize("enable_bias", [True, False])
    @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
    @pytest.mark.parametrize("float_data_type", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("int_data_type", [torch.int64])
    @pytest.mark.parametrize(
        "mask_type, target_group_size, max_num_context, max_num_target",
        [
            (MaskType.NONE, 0, 0, 0),
            (MaskType.CUSTOM, 0, 0, 0),
            (MaskType.TRIL, 1, 0, 30),
            (MaskType.TRIL, 3, 0, 30),
            (MaskType.TRIL, 1, 6, 0),
            (MaskType.TRIL, 3, 6, 0),
            (MaskType.TRIL, 1, 6, 30),
            (MaskType.TRIL, 3, 6, 30),
        ],
    )
    def test_hstu_jagged_deterministic_forward(
        self,
        batch_size,
        head_num,
        max_seq_len,
        head_dim,
        enable_bias,
        mask_type,
        silu_scale,
        float_data_type,
        int_data_type,
        target_group_size,
        max_num_context,
        max_num_target,
    ):
        qkv_shape_info = QKVShapeInfo(
            float_type=float_data_type,
            int_type=int_data_type,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_heads_q=head_num,
            num_heads_k=head_num,
            head_dim_qk=head_dim,
            head_dim_v=head_dim,
        )
        mask_info = MaskGenInfo(
            mask_type=mask_type,
            max_num_context=max_num_context,
            max_num_target=max_num_target,
            target_group_size=target_group_size,
        )
        self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale, deterministic=True)

    # fp8
    is_a5 = False
    if is_a5:

        @pytest.mark.parametrize("batch_size", [4, 128, 256])
        @pytest.mark.parametrize("head_num_q", [8])
        @pytest.mark.parametrize("head_num_k", [8, 4, 2, 1])
        @pytest.mark.parametrize("max_seq_len", [15, 256])
        @pytest.mark.parametrize("head_dim_qk", [64, 96, 128])
        @pytest.mark.parametrize("head_dim_v", [16, 32, 48, 64, 80])
        @pytest.mark.parametrize("enable_bias", [True, False])
        @pytest.mark.parametrize("silu_scale", [0, 1 / 1024])
        @pytest.mark.parametrize("float_data_type", [torch.float8_e4m3fn])
        @pytest.mark.parametrize("int_data_type", [torch.int32])
        @pytest.mark.parametrize(
            "mask_type, target_group_size, max_num_context, max_num_target",
            [
                (MaskType.NONE, 0, 0, 0),
                (MaskType.CUSTOM, 0, 0, 0),
                (MaskType.TRIL, 1, 0, 30),
                (MaskType.TRIL, 3, 0, 30),
                (MaskType.TRIL, 1, 6, 0),
                (MaskType.TRIL, 3, 6, 0),
                (MaskType.TRIL, 1, 6, 30),
                (MaskType.TRIL, 3, 6, 30),
            ],
        )
        def test_hstu_jagged_forward_fp8(
            self,
            batch_size,
            head_num_q,
            head_num_k,
            max_seq_len,
            head_dim_qk,
            head_dim_v,
            enable_bias,
            mask_type,
            silu_scale,
            float_data_type,
            int_data_type,
            target_group_size,
            max_num_context,
            max_num_target,
        ):
            qkv_shape_info = QKVShapeInfo(
                float_type=float_data_type,
                int_type=int_data_type,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_heads_q=head_num_q,
                num_heads_k=head_num_k,
                head_dim_qk=head_dim_qk,
                head_dim_v=head_dim_v,
            )
            mask_info = MaskGenInfo(
                mask_type=mask_type,
                max_num_context=max_num_context,
                max_num_target=max_num_target,
                target_group_size=target_group_size,
            )

            self.execute(qkv_shape_info, mask_info, enable_bias, silu_scale)
