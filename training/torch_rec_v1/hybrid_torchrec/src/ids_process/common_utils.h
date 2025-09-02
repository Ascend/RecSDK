/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef COMMON_UTILS_H
#define COMMON_UTILS_H

template <typename T>
T* GetSafeDataPtr(const at::Tensor& tensor, const char* message)
{
    TORCH_CHECK(tensor.defined() && tensor.numel() > 0, message, " is an empty tensor");

    TORCH_CHECK(tensor.dtype() == at::CppTypeToScalarType<T>::value, message, " tensor type mismatch");

    TORCH_CHECK(tensor.is_contiguous(), message, " must be contiguous");

    return tensor.data_ptr<T>();
}

#endif