/**
* Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"

at::Tensor gen_position_ids_reverse_v2_npu(const at::Tensor& seqlen, const at::Tensor& seqlenOffsets,
                                           const at::Tensor& rspos, int64_t batchSize,
                                           bool interleavedAction, bool withCtx)
{
    const at::OptionalDeviceGuard guard(device_of(seqlen));

    std::vector<at::Tensor> tensors = {seqlen, seqlenOffsets, rspos};
    std::vector<std::string> names = {"seqlen", "seqlen_offsets", "rspos"};
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(seqlen.dim() == 1, "seqlen should be 1D");
    TORCH_CHECK(seqlenOffsets.dim() == 1, "seqlenOffsets should be 1D");
    TORCH_CHECK(rspos.dim() == 1, "rspos should be 1D");

    TORCH_CHECK(seqlen.dtype() == at::kInt, "seqlen must be int32");
    TORCH_CHECK(seqlenOffsets.dtype() == at::kInt, "seqlenOffsets must be int32");
    TORCH_CHECK(rspos.dtype() == at::kInt, "rspos must be int32");

    TORCH_CHECK(batchSize >= 0, "batch_size must be non-negative");
    TORCH_CHECK(
            seqlen.size(0) == batchSize, "seqlen.size(0) must equal batch_size, got ", seqlen.size(0), " vs ", batchSize);
    TORCH_CHECK(seqlenOffsets.size(0) == batchSize + 1, "seqlenOffsets size must equal batchSize + 1");
    TORCH_CHECK(rspos.size(0) == batchSize, "rspos size must equal batchSize");

    TORCH_CHECK(!interleavedAction, "interleaved_action = true is not supported in NPU implementation");
    TORCH_CHECK(!withCtx, "with_ctx = true is not supported in NPU implementation");

    int64_t totalSeqLen = seqlenOffsets[batchSize].item<int64_t>();
    TORCH_CHECK(totalSeqLen >= 0, "seqlenOffsets[batch_size] must be non-negative");
    at::Tensor positionIds = at::empty({totalSeqLen}, seqlen.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnGenPositionIdsReverseV2,
                 seqlen,
                 seqlenOffsets,
                 rspos,
                 batchSize,
                 positionIds);
    return positionIds;
}

TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("gen_position_ids_reverse_v2(Tensor seqlen, Tensor seqlen_offsets, Tensor rspos, "
          "int batch_size, bool interleaved_action=False, bool with_ctx=False) -> Tensor");
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("gen_position_ids_reverse_v2(Tensor seqlen, Tensor seqlen_offsets, Tensor rspos, "
          "int batch_size, bool interleaved_action=False, bool with_ctx=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("gen_position_ids_reverse_v2", &gen_position_ids_reverse_v2_npu);
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("gen_position_ids_reverse_v2", &gen_position_ids_reverse_v2_npu);
}