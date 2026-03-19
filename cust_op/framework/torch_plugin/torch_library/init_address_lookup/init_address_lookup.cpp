/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <torch/library.h>

#include "../common/pytorch_npu_helper.hpp"
#include "../common/common_utils.h"

using namespace at;

/**
 * @brief 初始化地址查找表（FBGEMM 调用方式）
 *
 * 用于训练中嵌入剪枝。为每个嵌入表建立地址映射：
 * - 在 buffer_offsets[i] 到 buffer_offsets[i+1] 范围内
 * - 如果行索引 r < emb_sizes[i]，则映射到自身（address_lookup[r] = r）
 * - 否则映射到0（address_lookup[r] = 0）
 *
 * @param address_lookups 调用方预分配的地址查找表缓冲区（inplace写入），shape: [buffer_offsets[-1]]，类型: int64 或
 * int32
 * @param buffer_offsets  CSR格式的行偏移，定义每个表的起始索引，shape: [num_tables + 1]，类型: int64
 * @param emb_sizes       每个嵌入表的逻辑行数，shape: [num_tables]，类型: int64 或 int32
 * @note 该接口为 inplace 写入，返回值为 void；调用后结果写入 address_lookups
 */
void init_address_lookup_impl_npu(at::Tensor& address_lookups, at::Tensor& buffer_offsets, at::Tensor& emb_sizes)
{
    // totalRows == 0：无需计算，直接返回空 tensor
    if (address_lookups.numel() == 0) {
        return;
    }
    check_tensor_non_empty(buffer_offsets, "buffer_offsets");
    check_tensor_non_empty(emb_sizes, "emb_sizes");

    // 检查NPU设备且设备ID一致
    std::vector<at::Tensor> tensors = {address_lookups, buffer_offsets, emb_sizes};
    std::vector<std::string> names = {"address_lookups", "buffer_offsets", "emb_sizes"};
    check_tensor_npu_device(tensors, names);

    TORCH_CHECK(address_lookups.dim() == 1, "address_lookups should be 1D tensor");
    TORCH_CHECK(buffer_offsets.dim() == 1, "buffer_offsets should be 1D tensor");
    TORCH_CHECK(emb_sizes.dim() == 1, "emb_sizes should be 1D tensor");
    TORCH_CHECK(buffer_offsets.scalar_type() == at::kLong, "buffer_offsets should be int64 type");
    TORCH_CHECK(emb_sizes.scalar_type() == at::kLong || emb_sizes.scalar_type() == at::kInt,
                "emb_sizes should be int64 or int32 type");
    TORCH_CHECK(address_lookups.scalar_type() == emb_sizes.scalar_type(),
                "address_lookups and emb_sizes should have the same dtype");
    TORCH_CHECK(buffer_offsets.size(0) == emb_sizes.size(0) + 1,
                "buffer_offsets.size() should be emb_sizes.size() + 1");

    // 校验 address_lookups 的长度必须等于 buffer_offsets[-1]（总行数）
    int64_t expected_total_rows = buffer_offsets[buffer_offsets.size(0) - 1].item<int64_t>();
    TORCH_CHECK(address_lookups.numel() == expected_total_rows, "address_lookups.numel() (", address_lookups.numel(),
                ") must equal buffer_offsets[-1] (", expected_total_rows, ")");

    auto address_lookups_conti = address_lookups.contiguous();
    auto buffer_offsets_conti = buffer_offsets.contiguous();
    auto emb_sizes_conti = emb_sizes.contiguous();

    // 调用NPU算子（inplace方式，输入和输出是同一个tensor）
    EXEC_NPU_CMD(aclnnInitAddressLookup, buffer_offsets_conti, emb_sizes_conti, address_lookups_conti,
                 address_lookups_conti);

    // 若 contiguous() 产生了新 tensor（非连续内存），需拷贝回原 tensor
    if (!address_lookups_conti.is_same(address_lookups)) {
        address_lookups.copy_(address_lookups_conti);
    }

    return;
}

// 在mxrec命名空间里注册init_address_lookup schema
TORCH_LIBRARY_FRAGMENT(mxrec, m)
{
    m.def("init_address_lookup(Tensor address_lookups, Tensor buffer_offsets, Tensor emb_sizes) -> ()");
}

// fbgemm的schema只在GPU侧，故需注册
TORCH_LIBRARY_FRAGMENT(fbgemm, m)
{
    m.def("init_address_lookup(Tensor address_lookups, Tensor buffer_offsets, Tensor emb_sizes) -> ()");
}

// 为NPU设备注册实现
// NPU设备在pytorch 2.1及以上版本使用的设备名称是PrivateUse1
TORCH_LIBRARY_IMPL(mxrec, PrivateUse1, m)
{
    m.impl("init_address_lookup", &init_address_lookup_impl_npu);
}

TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m)
{
    m.impl("init_address_lookup", &init_address_lookup_impl_npu);
}
