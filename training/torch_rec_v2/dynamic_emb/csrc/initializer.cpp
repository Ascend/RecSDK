/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "initializer.h"

#include <iostream>
#include <cfloat>

#include "initializer_kernel_ops.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_utils.h"

#ifndef LOG_ERROR
#define LOG_ERROR(msg) std::cout << "[INFO]" << msg << std::endl
#endif

namespace dyn_emb {

namespace {

DataType to_data_type(at::ScalarType scalar_type)
{
    return scalartype_to_datatype(scalar_type);
}

struct IndexInitLaunchArgs {
    DataType value_type;
    DataType index_type;
    int64_t num;
    int64_t dim;
    int64_t stride;
    void* buffer;
    void* indices;
    aclrtStream stream;
};

IndexInitLaunchArgs make_index_init_launch_args(const at::Tensor& buffer, const at::Tensor& indices)
{
    IndexInitLaunchArgs args;
    args.value_type = to_data_type(buffer.scalar_type());
    args.index_type = to_data_type(indices.scalar_type());
    args.num = indices.size(0);
    args.dim = buffer.size(1);
    args.stride = buffer.stride(0);
    args.buffer = buffer.data_ptr();
    args.indices = indices.data_ptr();
    args.stream = c10_npu::getCurrentNPUStream().stream(true);
    return args;
}

template <typename LaunchFn>
void dispatch_index_init(const at::Tensor& buffer, const at::Tensor& indices, LaunchFn launch)
{
    if (indices.numel() == 0) {
        LOG_ERROR("Indices tensor is empty, initialization failed.");
        return;
    }
    if (buffer.dim() != 2) {
        LOG_ERROR("Initializer input buffer must be 2D.");
        return;
    }
    if (buffer.stride(1) != 1) {
        LOG_ERROR("Initializer input buffer must be contiguous at dim 1.");
        return;
    }
    auto args = make_index_init_launch_args(buffer, indices);
    launch(args);
}

bool validate_init_bounds(float lower, float upper)
{
    if (lower - upper > FLT_EPSILON) {
        LOG_ERROR("Lower bound must be less than upper bound.");
        return false;
    }
    return true;
}

}  // namespace

CurandStateContext::CurandStateContext()
{
    auto stream = c10_npu::getCurrentNPUStream().stream(true);
    alloc_curand_states(&states_, stream);
}

CurandStateContext::~CurandStateContext()
{
    free_curand_states(states_);
    states_ = nullptr;
}

curandState* CurandStateContext::ptr()
{
    return states_;
}

void normal_init(at::Tensor buffer, at::Tensor indices, CurandStateContext& curand_state_context, float mean,
                 float std_dev)
{
    dispatch_index_init(buffer, indices, [&](const IndexInitLaunchArgs& args) {
        launch_index_normal_init(args.value_type, args.index_type, args.num, args.dim, args.stride, args.buffer,
                                 args.indices, curand_state_context.ptr(), mean, std_dev, args.stream);
    });
}

void truncated_normal_init(at::Tensor buffer, at::Tensor indices, CurandStateContext& curand_state_context, float mean,
                           float std_dev, float lower, float upper)
{
    if (!validate_init_bounds(lower, upper)) {
        return;
    }
    dispatch_index_init(buffer, indices, [&](const IndexInitLaunchArgs& args) {
        launch_index_truncated_normal_init(args.value_type, args.index_type, args.num, args.dim, args.stride,
                                           args.buffer, args.indices, curand_state_context.ptr(), mean, std_dev, lower,
                                           upper, args.stream);
    });
}

void uniform_init(at::Tensor buffer, at::Tensor indices, CurandStateContext& curand_state_context, float lower,
                  float upper)
{
    if (!validate_init_bounds(lower, upper)) {
        return;
    }
    dispatch_index_init(buffer, indices, [&](const IndexInitLaunchArgs& args) {
        launch_index_uniform_init(args.value_type, args.index_type, args.num, args.dim, args.stride, args.buffer,
                                  args.indices, curand_state_context.ptr(), lower, upper, args.stream);
    });
}

void const_init(at::Tensor buffer, at::Tensor indices, float value)
{
    dispatch_index_init(buffer, indices, [&](const IndexInitLaunchArgs& args) {
        launch_index_const_init(args.value_type, args.index_type, args.num, args.dim, args.stride, args.buffer,
                                args.indices, value, args.stream);
    });
}

void debug_init(at::Tensor buffer, at::Tensor indices, at::Tensor keys)
{
    dispatch_index_init(buffer, indices, [&](const IndexInitLaunchArgs& args) {
        launch_index_debug_init(args.value_type, args.index_type, to_data_type(keys.scalar_type()), args.num, args.dim,
                                args.stride, args.buffer, args.indices, keys.data_ptr(), 100000, args.stream);
    });
}

}  // namespace dyn_emb
