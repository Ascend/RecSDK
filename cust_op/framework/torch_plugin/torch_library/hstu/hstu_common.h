/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

#ifndef HSTU_COMMON_H
#define HSTU_COMMON_H

#include <string>
#include <algorithm>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::AutogradContext;
using torch::autograd::Function;
using tensor_list = std::vector<at::Tensor>;
using namespace at;

constexpr size_t MIN_SEQ_LEN = 1;
constexpr size_t MAX_SEQ_LEN = 20480;
constexpr uint32_t MASK_TYPE_TRIL = 0;
constexpr uint32_t MASK_TYPE_TRIU = 1;
constexpr uint32_t MASK_TYPE_CUSTOM = 3;
constexpr uint32_t CONST_4 = 4;
constexpr uint32_t CONST_3 = 3;
constexpr uint32_t CONST_2 = 2;

namespace hstu {
inline bool MaskCheck(int64_t maskType, uint32_t maskIsDefine)
{
    TORCH_CHECK(MASK_TYPE_TRIL <= maskType <= MASK_TYPE_CUSTOM, "maskType expect in [0, 3], but value is ", maskType);
    TORCH_CHECK(maskType != MASK_TYPE_TRIU, "maskType current not support triu now, pls use custome mask");
    TORCH_CHECK(!(maskType == MASK_TYPE_CUSTOM && !maskIsDefine), "use custome mask must have valide mask tensor");
    return true;
}

inline bool MaxSeqLenCheck(int64_t maxSeqLen)
{
    TORCH_CHECK(maxSeqLen >= MIN_SEQ_LEN && maxSeqLen <= MAX_SEQ_LEN,
                "maxSeqLen expect in [1, 20480], but value is ", maxSeqLen);
    return true;
}

}
#endif  // HSTU_COMMON_H
