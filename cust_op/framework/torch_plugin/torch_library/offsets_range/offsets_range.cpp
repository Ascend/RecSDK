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
==============================================================================*/

#include <climits>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"
using tensor_list = std::vector<at::Tensor>;
using namespace at;

constexpr int64_t MAX_OFFSETS_LEN = 1LL << 17;
constexpr int64_t MAX_RANGE_SIZE_INT32 = INT32_MAX;
constexpr int64_t MAX_RANGE_SIZE_INT64 = LLONG_MAX;

// 为NPU设备注册前向实现
at::Tensor offsets_range_impl_npu(const at::Tensor& offsets, int64_t range_size)
{
    check_tensor_non_empty(offsets, "offsets");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {offsets};
    std::vector<std::string> names = {"offsets"};
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(offsets.dim() == 1, "The offsets should be 1D");
    TORCH_CHECK(offsets.scalar_type() == at::kInt || offsets.scalar_type() == at::kLong,
                "offsets must have int32 or int64 dtype.");

    const int64_t offsets_len = offsets.size(0);
    TORCH_CHECK(offsets_len >= 1 && offsets_len <= MAX_OFFSETS_LEN, "offsets length must be in [1, ", MAX_OFFSETS_LEN,
                "], got ", offsets_len);
#if defined(NPU_CHIP_A5) && NPU_CHIP_A5
    TORCH_CHECK(range_size >= 1 && range_size <= MAX_RANGE_SIZE_INT64, "range_size must be in [1, ",
                MAX_RANGE_SIZE_INT64, "], got ", range_size);
#else
    TORCH_CHECK(range_size >= 1 && range_size <= MAX_RANGE_SIZE_INT32, "range_size must be in [1, ",
                MAX_RANGE_SIZE_INT32, "], got ", range_size);
#endif

    auto offsets_conti = offsets.contiguous();
    at::Tensor result = at::empty(range_size, offsets_conti.options());
    EXEC_NPU_CMD(aclnnOffsetsRange, offsets_conti, range_size, result);
    return result;
}

// 在npu命名空间里注册offsets_range
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("offsets_range(Tensor offsets, int range_size) -> Tensor");
}

// 为NPU设备注册前向实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1，在2.1以下版本用的是XLA，如果是2.1以下版本PrivateUse1需要改成XLA
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("offsets_range", &offsets_range_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("offsets_range", &offsets_range_impl_npu);
}
