/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
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

#pragma once

#include "tiling/platform/platform_ascendc.h"
#include "aclnn/aclnn_base.h"
#include "acl/acl.h"

#include "utils.h"

#include <vector>
#include <cmath>
#include <cstdio>

namespace dyn_emb {
inline void CheckAclRet(bool cond, const std::string& msg)
{
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

aclTensor* CreateAclTensorFromAtTensor(const at::Tensor& tensor, aclDataType type)
{
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous for ACL");
    const auto& shape = tensor.sizes();
    const auto& strides = tensor.strides();
    aclDataType dataType = type;
    aclFormat format = ACL_FORMAT_ND;
    void* devicePtr = tensor.data_ptr();

    std::vector<int64_t> aclShape(shape.begin(), shape.end());
    std::vector<int64_t> aclStrides(strides.begin(), strides.end());

    aclTensor* aclTensorPtr = aclCreateTensor(aclShape.data(), aclShape.size(), dataType, aclStrides.data(), 0, format,
                                              aclShape.data(), aclShape.size(), devicePtr);
    TORCH_CHECK(aclTensorPtr != nullptr, "Failed to create aclTensor from at::Tensor");
    return aclTensorPtr;
}

class AclSingleton {
public:
    AclSingleton(const AclSingleton&) = delete;
    AclSingleton& operator=(const AclSingleton&) = delete;

    static AclSingleton& GetInstance()
    {
        static AclSingleton instance;
        return instance;
    }

    size_t GetMaxCores() const
    {
        return maxCores;
    }

private:
    AclSingleton()
    {
        const char* socVersion = SOC_VERSION;
        auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
        if (ascendcPlatform != nullptr) {
            maxCores = ascendcPlatform->GetCoreNumAiv();
        } else {
            throw std::runtime_error("ascendcPlatform for socVersion " + std::string(socVersion) + " does not exist");
        }
    }
    ~AclSingleton() {}

    size_t maxCores;
};
}  // namespace dyn_emb