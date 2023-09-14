/*
* Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
* Description: common module
* Author: MindX SDK
* Date: 2022/11/15
*/

#ifndef MX_REC_HD_TRANSFER_H
#define MX_REC_HD_TRANSFER_H

#include "acl/acl_base.h"
#include "acl/acl.h"
#include "acl/acl_tdt.h"
#include "acl/acl_tdt_queue.h"
#include "acl_channel.h"
#include "utils/common.h"
#include "utils/config.h"

#ifndef tdtCreateChannel
#define tdtCreateChannel acltdtCreateChannelWithCapacity
#endif

namespace MxRec {
    using namespace std;
    const std::string MGMT = "\033[32m[Mgmt]\033[0m ";
    const std::string HD = "\033[32m[HD]\033[0m ";
    const std::string HOSTEMB = "\033[32m[HostEmb]\033[0m ";
    const int PING_PONG_SIZE = 6;

    enum class TransferChannel {
        D2H,
        RESTORE,
        RESTORE_SECOND,
        ALL2ALL,
        UNIQKEYS,
        LOOKUP,
        EVICT,
        H2D,
        SWAP,
        INVALID
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
            case TransferChannel::EVICT:
                return "evict";
            case TransferChannel::H2D:
                return "h2d";
            case TransferChannel::SWAP:
                return "swap";
            default:
                throw std::invalid_argument("Invalid TransferChannel");
        }
    };

    class HDTransfer {
    public:
        std::unordered_map<std::string, acltdtDataset*> aclDatasets;

        HDTransfer() = default;

        int Init(const vector<EmbInfo>& embInfos, uint32_t localRankId);

        void Send(TransferChannel channel, const vector<Tensor>& tensors,
                  int channelId, const string& embName, int batchId = -1);

        vector<Tensor> Recv(TransferChannel channel, int channelId, const string& embName);

        size_t RecvAcl(TransferChannel channel, int channelId, const string& embName);

        void Destroy();

    private:
#ifndef GTEST
        std::unordered_map<std::string, acltdtChannelHandle*> transferChannels;
#endif
        bool running;
        void CreateChannel(const uint32_t localRankId, const string& embName, const int channelNum);
    };
}
#endif // MX_REC_HD_TRANSFER_H
