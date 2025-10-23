/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

#include "hd_transfer/hd_transfer.h"

namespace MxRec {
class HDTransferMock : public HDTransfer {
public:
    MOCK_METHOD4(Init, int(const std::vector<EmbInfo>& embInfos, uint32_t localRankId,
                           bool isIncrementalCkpt, bool useLccl));

    MOCK_METHOD5(Send, void(TransferChannel channel, const std::vector<Tensor>& tensors,
                            int channelId, const std::string& embName, int batchId));

    MOCK_METHOD5(RecvAcl, size_t(TransferChannel channel, int channelId,
                                 const std::string& embName, int embeddingThreadId, int batchId));

    MOCK_METHOD3(RecvOffsetsAcl, size_t(TransferChannel channel, int channelId, const std::string& embName));

    MOCK_METHOD0(Destroy, void());

    MOCK_METHOD0(GetTransChannel, std::unordered_map<std::string, acltdtChannelHandle*>());

    MOCK_METHOD0(GetUsedTransChannel, std::unordered_map<int, std::set<std::string>>());

    MOCK_METHOD1(ClearTransChannel, void(int channelId));
};
} //  namespace MxRec