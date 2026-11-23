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

constexpr float DEFAULT_TIME_SCALE = 300.0f;

at::Tensor gen_position_ids_with_timestamp_npu(const at::Tensor& seqlen, const at::Tensor& seqlenOffsets,
                                               const at::Tensor& timestamps, const int64_t batchSize,
                                               const int64_t totalSeqLen, const c10::optional<double>& timeScale)
{
    const at::OptionalDeviceGuard guard(device_of(seqlen));

    // 检查输入张量在 NPU 设备上
    std::vector<at::Tensor> tensors = {seqlen, seqlenOffsets, timestamps};
    std::vector<std::string> names = {"seqlen", "seqlen_offsets", "timestamps"};
    check_tensor_npu_device(tensors, names);

    // 参数校验
    TORCH_CHECK(seqlen.dim() == 1, "The seqlen should be 1D");
    TORCH_CHECK(seqlenOffsets.dim() == 1, "The seqlenOffsets should be 1D");
    TORCH_CHECK(timestamps.dim() == 1, "The timestamps should be 1D");

    TORCH_CHECK(seqlen.dtype() == at::kInt, "seqlen must be int32");
    TORCH_CHECK(seqlenOffsets.dtype() == at::kInt, "seqlenOffsets must be int32");
    TORCH_CHECK(timestamps.dtype() == at::kInt, "timestamps must be int32");

    TORCH_CHECK(seqlen.size(0) == batchSize, "seqlen size must equal batchSize");
    TORCH_CHECK(seqlenOffsets.size(0) == batchSize + 1, "seqlenOffsets size must equal batchSize + 1");
    TORCH_CHECK(timestamps.size(0) == totalSeqLen, "timestamps size must equal totalSeqLen");

    double realTimeScale = timeScale.value_or(DEFAULT_TIME_SCALE);
    TORCH_CHECK(realTimeScale > 0, "timeScale must be positive");

    at::Tensor positionIds = at::empty({totalSeqLen}, timestamps.options().dtype(at::kInt));

    EXEC_NPU_CMD(aclnnGenPositionIdsWithTimestamp, seqlen, seqlenOffsets, timestamps, realTimeScale, positionIds);
    return positionIds;
}

// 注册算子到 fbgemm 和 mxrec 命名空间
TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("gen_position_ids_with_timestamp("
          "Tensor seqlen, Tensor seqlen_offsets, Tensor timestamps, "
          "int batch_size, int total_seq_len, float? time_scale=300.0) -> Tensor");
}

TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("gen_position_ids_with_timestamp("
          "Tensor seqlen, Tensor seqlen_offsets, Tensor timestamps, "
          "int batch_size, int total_seq_len, float? time_scale=300.0) -> Tensor");
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("gen_position_ids_with_timestamp", &gen_position_ids_with_timestamp_npu);
}

TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("gen_position_ids_with_timestamp", &gen_position_ids_with_timestamp_npu);
}