/**
 * @file pytorch_npu_const.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef PYTORCH_NPU_CONST_H
#define PYTORCH_NPU_CONST_H

#include <ATen/Tensor.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <c10/util/Exception.h>
#include <dlfcn.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>

#include <functional>
#include <type_traits>
#include <vector>
#include <filesystem>
#include <fstream>

#include "securec.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/CalcuOpUtil.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

#define NPU_NAME_SPACE at_npu::native

typedef struct aclOpExecutor aclOpExecutor;
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclFloatArray aclFloatArray;
typedef struct aclBoolArray aclBoolArray;
typedef struct aclTensorList aclTensorList;

typedef aclTensor* (*_aclCreateTensor)(const int64_t* view_dims, uint64_t view_dims_num, aclDataType data_type,
                                       const int64_t* stride, int64_t offset, aclFormat format,
                                       const int64_t* storage_dims, uint64_t storage_dims_num, void* tensor_data);
typedef aclScalar* (*_aclCreateScalar)(void* value, aclDataType data_type);
typedef aclIntArray* (*_aclCreateIntArray)(const int64_t* value, uint64_t size);
typedef aclFloatArray* (*_aclCreateFloatArray)(const float* value, uint64_t size);
typedef aclBoolArray* (*_aclCreateBoolArray)(const bool* value, uint64_t size);
typedef aclTensorList* (*_aclCreateTensorList)(const aclTensor* const *value, uint64_t size);

typedef int (*_aclDestroyTensor)(const aclTensor* tensor);
typedef int (*_aclDestroyScalar)(const aclScalar* scalar);
typedef int (*_aclDestroyIntArray)(const aclIntArray* array);
typedef int (*_aclDestroyFloatArray)(const aclFloatArray* array);
typedef int (*_aclDestroyBoolArray)(const aclBoolArray* array);
typedef int (*_aclDestroyTensorList)(const aclTensorList* array);

constexpr int kHashBufSize = 8192;
constexpr int kHashBufMaxSize = kHashBufSize + 1024;
extern thread_local char g_hashBuf[kHashBufSize];
extern thread_local int g_hashOffset;

#endif