/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#pragma once

#include <ATen/Tensor.h>

/**
 * @brief 异步完成累积求和操作
 * @param offset 输入的一维张量
 * @return 累积求和结果张量
 */
at::Tensor asynchronous_complete_cumsum_npu(const at::Tensor &offset);
