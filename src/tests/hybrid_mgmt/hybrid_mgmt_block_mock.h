/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.s
See the License for the specific language governing permissions and
        limitations under the License.
==============================================================================*/

#pragma once

#include <gmock/gmock.h>

#include "hybrid_mgmt/hybrid_mgmt_block.h"

namespace MxRec {
class HybridMgmtBlockMock : public HybridMgmtBlock {
public:
    MOCK_METHOD1(CheckAndNotifyWake, void(int channelId));
    MOCK_METHOD2(CountPythonStep, void(int channelId, int steps));
    MOCK_METHOD1(CheckAndSetBlock, void(int channelId));
    MOCK_METHOD1(CheckValid, void(int channelId));
    MOCK_METHOD1(DoBlock, void(int channelId));
    MOCK_METHOD1(ResetAll, void(int channelId));
    MOCK_METHOD1(GetBlockStatus, bool(int channelId));
    MOCK_METHOD2(SetBlockStatus, void(int channelId, bool block));
    MOCK_METHOD1(SetRankInfo, void(RankInfo ri));
    MOCK_METHOD2(SetStepInterval, void(int trainStep, int evalStep));
    MOCK_METHOD1(WaitValid, bool(int channelId));
    MOCK_METHOD0(Destroy, void());
};
} // namespace MxRec