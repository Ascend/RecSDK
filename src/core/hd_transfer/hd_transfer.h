/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef MX_REC_HD_TRANSFER_H
#define MX_REC_HD_TRANSFER_H

#include <mutex>

#include "acl/acl_base.h"
#include "acl/acl.h"
#include "acl/acl_tdt.h"
#include "acl/acl_tdt_queue.h"
#include "acl_channel.h"
#include "utils/common.h"
#include "utils/config.h"

#ifndef TDT_CREATE_CHANNEL
#define TDT_CREATE_CHANNEL acltdtCreateChannelWithCapacity
#endif

namespace MxRec {
using namespace std;
const std::string MGMT = "\033[32m[Mgmt]\033[0m ";
const int PING_PONG_SIZE = 6;

enum class TransferChannel {
    D2H,
    RESTORE,
    RESTORE_SECOND,
    ALL2ALL,
    UNIQKEYS,
    LOOKUP,
    MASK,
    EVICT,
    H2D,
    SWAP,
    SAVE_D2H,
    SAVE_H2D,
    KEY_D2H,
    INVALID,
};

inline string TransferChannel2Str(TransferChannel e)
{
    switch (e) {
        case TransferChannel::RESTORE_SECOND:
            return "restore_second";
        case TransferChannel::D2H:
            return "d2h";
        case TransferChannel::RESTORE:
            return "restore";
        case TransferChannel::ALL2ALL:
            return "all2all";
        case TransferChannel::UNIQKEYS:
            return "uniquekeys";
        case TransferChannel::LOOKUP:
            return "lookup";
        case TransferChannel::MASK:
            return "mask";
        case TransferChannel::EVICT:
            return "evict";
        case TransferChannel::H2D:
            return "h2d";
        case TransferChannel::SWAP:
            return "swap";
        case TransferChannel::SAVE_D2H:
            return "save_d2h";
        case TransferChannel::SAVE_H2D:
            return "save_h2d";
        case TransferChannel::KEY_D2H:
            return "key_d2h";
        default:
            throw std::invalid_argument("Invalid TransferChannel");
    }
};

class HDTransfer {
public:
    std::unordered_map<std::string, std::unordered_map<int, acltdtDataset*>> aclDatasets;
    std::unordered_map<std::string, acltdtDataset*> aclDatasetsForIncrementalCkpt;

    HDTransfer() = default;

    int Init(const vector<EmbInfo>& embInfos, uint32_t localRankId, bool isIncrementalCkpt);

    void Send(TransferChannel channel, const vector<Tensor>& tensors, int channelId, const string& embName,
              int batchId = -1);

    size_t RecvAcl(TransferChannel channel, int channelId, const string& embName, int embeddingThreadId, int batchId);
    size_t RecvOffsetsAcl(TransferChannel channel, int channelId, const string& embName);

    void Destroy();

    std::unordered_map<std::string, acltdtChannelHandle*> GetTransChannel();

    unordered_map<int, set<std::string>> GetUsedTransChannel();

    void ClearTransChannel(int channelId);

private:
    void CreateChannel(const uint32_t localRankId, const string& embName, const int channelNum);
    void CreateChannelForIncrementalCkpt(const uint32_t localRankId, const string& embName, const int channelNum);
    void RecordTrainingChannelStr(TransferChannel channel, const int channelId);

    std::unordered_map<std::string, acltdtChannelHandle*> transferChannels;
    std::unordered_map<int, std::set<std::string>> usedChannelsNames;  // The key indicates channels 0 and 1.
    bool running;
    std::mutex recordChannelMtx;
};

}  // namespace MxRec

#endif  // MX_REC_HD_TRANSFER_H
