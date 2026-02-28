/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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


#ifndef TILING_POLICY_JAGGED_H
#define TILING_POLICY_JAGGED_H

#include "tiling_policy.h"
#include "hstu_jagged_forward_tiling.h"

namespace HstuJaggedForward {

class TilingPolicyJagged : public HstuForward::TilingPolicy {
public:
    ge::graphStatus TilingProcess(gert::TilingContext* context) override;
private:
    bool TilingAttribute(gert::TilingContext* context, optiling::HstuJaggedForwardTilingData& tiling);
    bool TilingShape(gert::TilingContext* context, optiling::HstuJaggedForwardTilingData& tiling);
    bool TilingKeySet(gert::TilingContext* context, optiling::HstuJaggedForwardTilingData& tiling);
    bool TilingSaveToBuffer(gert::TilingContext* context, optiling::HstuJaggedForwardTilingData& tiling);
};

}

#endif