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

#include <iostream>
#include <string>
#include <memory>

#include "tiling_policy_factory.h"

namespace HstuDenseForwardFuxi {

std::vector<std::shared_ptr<TilingPolicy>> TilingPolicyFactory::m_policyMap(TYPE_NUM, std::make_shared<TilingPolicy>());

void TilingPolicyFactory::TilingPolicyRegister(LAYOUT_TYPE layOutType, std::shared_ptr<TilingPolicy>policy)
{
    if (layOutType < LAYOUT_TYPE::INVALID && layOutType >= LAYOUT_TYPE::NORMAL) {
        m_policyMap[static_cast<int>(layOutType)] = policy;
    }
    return;
}

TilingPolicyFactory &TilingPolicyFactory::GetInstance()
{
    static TilingPolicyFactory instance;
    return instance;
}

std::shared_ptr<TilingPolicy> TilingPolicyFactory::CreatePolicy(const char *layOutCStr)
{
    OPS_LOG_E_IF_NULL("layOutCStr", layOutCStr, return nullptr);

    auto layout = TilingPolicyFactory::ParseLayout(layOutCStr);
    if (layout < LAYOUT_TYPE::INVALID && layout >= LAYOUT_TYPE::NORMAL) {
        return m_policyMap[static_cast<int>(layout)];
    } else {
        OPS_LOG_E("[ERROR]", "the input Layout should be normal/jagged, but got %s.", layOutCStr);
        // TilingPolicy wiil tiling failed inside
        return std::make_shared<TilingPolicy>();
    }
}

LAYOUT_TYPE TilingPolicyFactory::ParseLayout(const char *layOutCStr)
{
    std::string layoutStr = std::string(layOutCStr);

    // 忽略大小写 统一转成小写
    for (auto &c : layoutStr) {
        c = tolower(c);
    }

    LAYOUT_TYPE layout;
    if (layoutStr == "normal") {
        layout = LAYOUT_TYPE::NORMAL;
    } else if (layoutStr == "normalv200") {
        layout = LAYOUT_TYPE::NORMALV200;
    } else if (layoutStr == "jagged") {
        layout = LAYOUT_TYPE::JAGGED;
    } else {
        layout = LAYOUT_TYPE::INVALID;
    }
    return layout;
}
}