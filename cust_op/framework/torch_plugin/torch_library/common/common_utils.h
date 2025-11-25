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
#include <vector>
#include <algorithm>
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
    TORCH_CHECK(tensor.dim() > 0, name, " tensor must have non-zero dimensions");
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

/**
 * 检查张量设备是否为NPU且设备ID一致
 * @param tensors 张量列表
 * @param names 张量名称列表(用于错误信息)
 * @throw torch::library::Exception 如果deviceType不是NPU，或deviceId不一致
 */
inline void check_tensor_npu_device(const std::vector<at::Tensor>& tensors,
                                    const std::vector<std::string>& names)
{
    // 检查所有张量是否都在NPU设备上
    for (size_t i = 0; i < tensors.size(); ++i) {
        TORCH_CHECK(tensors[i].device().type() == at::kPrivateUse1,
                    names[i], " tensor must be on NPU device, but got device type: ",
                    static_cast<int>(tensors[i].device().type()));
    }
    
    // 如果只有一个张量，不需要检查设备ID一致性
    if (tensors.size() < 2) {
        return;
    }
    // 获取第一个张量的设备ID作为参考
    int64_t expected_device_id = tensors[0].device().index();
    
    // 检查所有张量的设备ID是否一致
    for (size_t i = 1; i < tensors.size(); ++i) {
        int64_t current_device_id = tensors[i].device().index();
        TORCH_CHECK(current_device_id == expected_device_id,
                    names[i], " device ID (", current_device_id,
                    ") must match ", names[0], " device ID (", expected_device_id, ")");
    }
}

class ShapeRange {
public:
    int64_t lbound{0};
    int64_t ubound{0};
    int64_t mutiple{0};
    const char* name{nullptr};
    ShapeRange(int64_t lbound, int64_t ubound, int64_t mutiple, const char* name)
    {
        this->lbound = lbound;
        this->ubound = ubound;
        this->mutiple = mutiple;
        this->name = name;
    }

    bool Check(int64_t val) const
    {
        if (val < lbound || val > ubound || val % mutiple != 0) {
            return false;
        }
        return true;
    }
};


inline bool CheckInList(int64_t val, const std::vector<int64_t>& validValues)
{
    return std::find(validValues.begin(), validValues.end(), val) != validValues.end();
}


/**
 * 检查参数列表长度是否符合预期
 * @param list_size 列表长度
 * @param expect_size 期望长度
 * @param msg 列表名称/描述
 * @throw torch::library::Exception 长度不符合预期时抛出异常
 */
inline void check_param_len(size_t list_size, size_t expect_size, const std::string& msg)
{
    TORCH_CHECK(list_size == expect_size, " size of param:", msg, " must be ", expect_size, ", but got ", list_size)
}

inline bool CheckOptionalTensorIsNotNone(const c10::optional<at::Tensor>& tensor)
{
    return tensor.has_value() && tensor.value().defined();
}
#endif // COMMON_UTILS_H
