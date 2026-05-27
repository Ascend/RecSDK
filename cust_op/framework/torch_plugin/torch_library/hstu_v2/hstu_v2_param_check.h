/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
============================================================================== */

#pragma once

#include <ATen/ATen.h>
#include "../common/common_utils.h"

namespace hstu_v2 {

class HstuV2ParamChecker {
public:
    static constexpr int64_t kMaxHeads = 16;
    static constexpr int64_t kDimAlign = 16;
    static constexpr int64_t kMaxDim = 256;

    HstuV2ParamChecker(const at::Tensor& grad, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                       const at::Tensor& seqOffsetQ, const at::Tensor& seqOffsetK, int64_t maxSeqLenQ,
                       int64_t maxSeqLenK, const c10::optional<at::Tensor>& rab,
                       const c10::optional<at::Tensor>& numContext, const c10::optional<at::Tensor>& numTarget,
                       const c10::optional<int64_t>& targetGroupSize)
    {
        CheckTensors(grad, q, k, v, seqOffsetQ, seqOffsetK);
        CheckAttrs(maxSeqLenQ, maxSeqLenK);
        CheckMask(numContext, numTarget, targetGroupSize);
        if (rab.has_value() && rab.value().defined()) {
            CheckRab(rab.value(), maxSeqLenQ, maxSeqLenK);
        }
    }

    // Validated dimensions
    int64_t batchSize = 0;
    int64_t heads = 0;
    int64_t dimQK = 0;
    int64_t dimGV = 0;
    int64_t totalSeqLenQ = 0;
    int64_t totalSeqLenK = 0;

private:
    void CheckTensors(const at::Tensor& grad, const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                      const at::Tensor& seqOffsetQ, const at::Tensor& seqOffsetK)
    {
        TORCH_CHECK(grad.dim() == 3 && q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "grad/q/k/v must be 3D");
        TORCH_CHECK(seqOffsetQ.dim() == 1 && seqOffsetK.dim() == 1, "seq_offset must be 1D");

        totalSeqLenQ = q.size(0);
        heads = q.size(1);
        dimQK = q.size(2);
        totalSeqLenK = k.size(0);
        dimGV = v.size(2);
        batchSize = seqOffsetQ.size(0) - 1;

        TORCH_CHECK(grad.size(0) == totalSeqLenQ && grad.size(1) == heads && grad.size(2) == dimGV,
                    "grad shape mismatch, expect [", totalSeqLenQ, ",", heads, ",", dimGV, "], got [", grad.size(0),
                    ",", grad.size(1), ",", grad.size(2), "]");
        TORCH_CHECK(k.size(1) == heads && k.size(2) == dimQK, "k shape mismatch, expect heads=", heads,
                    " dimQK=", dimQK, ", got heads=", k.size(1), " dim=", k.size(2));
        TORCH_CHECK(v.size(0) == totalSeqLenK && v.size(1) == heads && v.size(2) == dimGV, "v shape mismatch, expect [",
                    totalSeqLenK, ",", heads, ",", dimGV, "], got [", v.size(0), ",", v.size(1), ",", v.size(2), "]");
        TORCH_CHECK(seqOffsetK.size(0) == seqOffsetQ.size(0), "seq_offset size mismatch, got ", seqOffsetK.size(0),
                    " vs ", seqOffsetQ.size(0));

        TORCH_CHECK(batchSize > 0, "batchSize must be > 0, got ", batchSize);
        TORCH_CHECK(heads >= 1 && heads <= kMaxHeads, "heads must be in [1,", kMaxHeads, "], got ", heads);
        TORCH_CHECK(dimQK % kDimAlign == 0 && dimQK <= kMaxDim, "dimQK must be multiple of ", kDimAlign,
                    " and <= ", kMaxDim, ", got ", dimQK);
        TORCH_CHECK(dimGV % kDimAlign == 0 && dimGV <= kMaxDim, "dimGV must be multiple of ", kDimAlign,
                    " and <= ", kMaxDim, ", got ", dimGV);

        const auto dtype = grad.scalar_type();
        TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "grad/q/k/v must be float16 or bfloat16, got ",
                    c10::toString(dtype));
        TORCH_CHECK(q.scalar_type() == dtype && k.scalar_type() == dtype && v.scalar_type() == dtype,
                    "grad/q/k/v must have the same dtype, got grad:", c10::toString(dtype),
                    " q:", c10::toString(q.scalar_type()), " k:", c10::toString(k.scalar_type()),
                    " v:", c10::toString(v.scalar_type()));

        TORCH_CHECK(seqOffsetQ.scalar_type() == at::kInt, "seqOffsetQ must be int32, got ",
                    c10::toString(seqOffsetQ.scalar_type()));
        TORCH_CHECK(seqOffsetK.scalar_type() == at::kInt, "seqOffsetK must be int32, got ",
                    c10::toString(seqOffsetK.scalar_type()));
    }

    void CheckAttrs(int64_t maxSeqLenQ, int64_t maxSeqLenK)
    {
        TORCH_CHECK(maxSeqLenQ > 0, "maxSeqLenQ must be > 0, got ", maxSeqLenQ);
        TORCH_CHECK(maxSeqLenK > 0, "maxSeqLenK must be > 0, got ", maxSeqLenK);
    }

    void CheckMask(const c10::optional<at::Tensor>& numContext, const c10::optional<at::Tensor>& numTarget,
                   const c10::optional<int64_t>& targetGroupSize)
    {
        bool hasCtx = CheckOptionalTensorIsNotNone(numContext);
        bool hasTgt = CheckOptionalTensorIsNotNone(numTarget);

        bool IsFullMask = !hasCtx && !hasTgt;
        TORCH_CHECK(IsFullMask, "current only support full mask.");
    }

    void CheckRab(const at::Tensor& rab, int64_t maxSeqLenQ, int64_t maxSeqLenK)
    {
        TORCH_CHECK(rab.dim() == 4, "rab must be 4D, got ", rab.dim(), "D");
        TORCH_CHECK(
            rab.size(0) == batchSize && rab.size(1) == heads && rab.size(2) == maxSeqLenQ && rab.size(3) == maxSeqLenK,
            "rab shape mismatch, expect [", batchSize, ",", heads, ",", maxSeqLenQ, ",", maxSeqLenK, "], got [",
            rab.size(0), ",", rab.size(1), ",", rab.size(2), ",", rab.size(3), "]");
        TORCH_CHECK(rab.scalar_type() == at::kHalf || rab.scalar_type() == at::kBFloat16,
                    "rab must be float16 or bfloat16, got ", c10::toString(rab.scalar_type()));
    }
};

}  // namespace hstu_v2
