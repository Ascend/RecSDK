/**
 * @file common_utils.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

#include <ATen/ATen.h>
#include <string>

/**
 * @file common_utils.h
 * @brief 常用张量检查工具函数
 * @note 该头文件需要与PyTorch ATen库一起使用
 */

/**
 * 检查张量是否非空
 * @param tensor 要检查的张量
 * @param name 张量名称(用于错误信息)
 * @throw torch::library::Exception 如果张量未定义或为空
 */
inline void check_tensor_non_empty(const at::Tensor& tensor, const std::string& name)
{
    TORCH_CHECK(tensor.defined(), name, " tensor must be defined");
    TORCH_CHECK(tensor.numel() > 0, name, " tensor must be non-empty");
}

/**
 * 检查张量维度是否符合预期
 * @param tensor 要检查的张量
 * @param expectedDim 期望的维度
 * @param name 张量名称(用于错误信息)
 * @throw torch::library::Exception 如果张量维度不符合预期
 */
inline void check_tensor_dim(const at::Tensor& tensor, int64_t expectedDim, const std::string& name)
{
    TORCH_CHECK(tensor.dim() == expectedDim, name, " must be ", expectedDim, "D");
}

#endif // COMMON_UTILS_H
