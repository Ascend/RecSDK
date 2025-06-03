/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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


#ifndef TILING_POLICY_FACTORY
#define TILING_POLICY_FACTORY

#include "tiling_policy.h"

namespace HstuDenseForward {

const int TYPE_NUM = 3;

enum class LAYOUT_TYPE {
    NORMAL = 0,
    NORMALV200 = 1,
    JAGGED = 2,
    INVALID = 3
};

class TilingPolicyFactory {
public:
    static std::shared_ptr<TilingPolicy> CreatePolicy(const char *layOutCStr);
    static void TilingPolicyRegister(LAYOUT_TYPE policyKey, std::shared_ptr<TilingPolicy> policy);
    static TilingPolicyFactory &GetInstance();
private:
    static LAYOUT_TYPE ParseLayout(const char *layOutCStr);
    static std::vector<std::shared_ptr<TilingPolicy>> m_policyMap;
};

class RegisterPolicy {
public:
    RegisterPolicy(LAYOUT_TYPE policyKey, std::shared_ptr<TilingPolicy> policy)
    {
        TilingPolicyFactory::GetInstance().TilingPolicyRegister(policyKey, policy);
    }
    ~RegisterPolicy()=default;
};

#define REGISTER_POLICY(policyKey, policy) \
    static RegisterPolicy g_##po(policyKey, policy);
}

#endif