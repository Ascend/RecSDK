/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
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

#include "utils.h"
#include <stdexcept>
#include <acl/acl.h>
namespace dyn_emb {
// 当前无接口获取每个block下最大线程数，查询手册Ascend950PR该值为2048
constexpr uint32_t MAX_THREADS_PER_BLOCK = 2048;

DeviceProp& DeviceProp::getDeviceProp(int device_id)
{
    static DeviceProp device_prop(device_id);
    return device_prop;
}

DeviceProp::DeviceProp(int device_id)
{
    uint32_t deviceCount = 0;
    if (aclrtGetDeviceCount(&deviceCount) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtGetDeviceCount failed");
    }

    if (deviceCount == 0 || device_id < 0 || device_id >= deviceCount) {
        throw std::runtime_error("Can't get device count, or device_id < 0, or device_id >= deviceCount, device_id = " +
            std::to_string(device_id) + ", deviceCount = " + std::to_string(deviceCount));
    }

    int64_t vectorCoreCount = 0;
    if (aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCoreCount) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtGetDeviceInfo(VECTOR_CORE_NUM) failed");
    }
    this->num_sms = vectorCoreCount;

    int64_t warpSize = 0;
    if (aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_WARP_SIZE, &warpSize) != ACL_SUCCESS) {
        throw std::runtime_error("aclrtGetDeviceInfo(WARP_SIZE) failed");
    }
    this->warp_size = warpSize;

    int64_t maxThreadPerVectorCore = 0;
    if (aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE, &maxThreadPerVectorCore) !=
        ACL_SUCCESS) {
        throw std::runtime_error("aclrtGetDeviceInfo(MAX_THREAD_PER_VECTOR_CORE) failed");
    }
    this->max_thread_per_sm = maxThreadPerVectorCore;
    this->max_thread_per_block = MAX_THREADS_PER_BLOCK;
    this->total_threads = this->num_sms * this->max_thread_per_sm;
}
}  // namespace dyn_emb