/*
* Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
* Description: common module
* Author: MindX SDK
* Date: 2022/11/15
*/
#include "hd_transfer.h"
#include <fstream>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/bundled/ranges.h>
#include "utils/common.h"

using namespace MxRec;
using namespace std;

int HDTransfer::Init(const vector<EmbInfo>& embInfos, uint32_t localRankId)
{
#ifndef GTEST
    spdlog::info(MGMT + "begin hd_transfer initialize, rank:{}", localRankId);
    aclError retOk = aclInit(nullptr);
    spdlog::info(MGMT + "end aclInit, rank:{}", localRankId);
    if (retOk != ACL_SUCCESS) {
        spdlog::error(MGMT + "aclInit fail, rank:{}, errno:{}", localRankId, retOk);
        return false;
    }
    spdlog::info(MGMT + "start Set device, rank:{}", localRankId);
    auto ret = aclrtSetDevice(static_cast<int32_t>(localRankId));
    if (ret != ACL_ERROR_NONE) {
        spdlog::error("Set device failed, device_id:{}", localRankId);
        return false;
    }
    spdlog::info(MGMT + "end Set device, rank:{}", localRankId);
    for (const auto& embInfo: embInfos) {
        vector<string> names = {embInfo.name};
        if (embInfo.modifyGraph) {
            names = embInfo.channelNames;
        }
        for (const string& name: names) {
            for (int i = 0; i < MAX_CHANNEL_NUM; ++i) {
                CreateChannel(localRankId, name, i);
            }
        }
    }
    running = true;
    spdlog::info("hd_transfer init");
#endif
    return true;
}

void HDTransfer::Destroy()
{
#ifndef GTEST
    running = false;
    spdlog::info(HD + "destroy channel start");
    for (auto& c: transferChannels) {
        tensorflow::StopRecvTensorByAcl(&c.second, c.first);
        spdlog::info(HD + "destroy channel:{}", c.first);
    }
    aclFinalize();
#endif
}

void HDTransfer::CreateChannel(uint32_t localRankId, const string& embName, int channelNum)
{
#ifndef GTEST
    int channelSize;
    const char* env = getenv("HD_CHANNEL_SIZE");
    if (env == nullptr) {
        channelSize = LARGE_CHANNEL_SIZE;
    } else {
        try {
            channelSize = stoi(env);
        } catch (const std::invalid_argument& e) {
            spdlog::warn("wrong HD_CHANNEL_SIZE env {}", e.what());
            channelSize = LARGE_CHANNEL_SIZE;
        } catch (const std::out_of_range& e) {
            spdlog::warn("wrong HD_CHANNEL_SIZE env {}", e.what());
            channelSize = LARGE_CHANNEL_SIZE;
        }
        if (channelSize <= 0) {
            channelSize = LARGE_CHANNEL_SIZE;
        }
    }
    spdlog::info("user config all2all restore lookup channel size:{}", channelSize);
    for (int c = D2H; c != INVALID; c++) {
        auto channel = static_cast<TransferChannel>(c);
        string sendName = fmt::format("{}_{}_{}", embName, TransferChannel2Str(channel), channelNum);
        if (TransferChannel2Str(channel) == "all2all" ||
            TransferChannel2Str(channel) == "restore" ||
            TransferChannel2Str(channel) == "lookup"  ||
            TransferChannel2Str(channel) == "evict"  /* for noDDR */
                ) {
            transferChannels[sendName] = tdtCreateChannel(localRankId, sendName.c_str(), channelSize);
        } else {
            transferChannels[sendName] = tdtCreateChannel(localRankId, sendName.c_str(), PING_PONG_SIZE);
        }
        spdlog::info("create channel:{} {}", sendName, static_cast<void*>(transferChannels[sendName]));
    }
#endif
}

void HDTransfer::Send(TransferChannel channel, const vector<Tensor> &tensors, int channelId, const string &embName,
                      int batchId)
{
    EASY_FUNCTION()
    if (!running) {
        return;
    }
#ifndef GTEST
    vector<size_t> sizes;
    for (auto& t: tensors) {
        sizes.push_back(t.NumElements());
    }
    string sendName = fmt::format("{}_{}_{}", embName, TransferChannel2Str(channel), channelId);

    spdlog::info(HD + "hd transfer send {}, send count is {}, size list:{}", sendName, sizes.size(),
                 sizes);

    if (sizes.size() == 0) {
        spdlog::warn("tensors num can not be zero");
        return;
    }
    bool isNeedResend = false;
    int resendTime = 0;
    tensorflow::Status status = tensorflow::Status::OK();
    do {
        status = tensorflow::SendTensorsByAcl(transferChannels[sendName], ACL_TENSOR_DATA_TENSOR, tensors,
                                              isNeedResend);
        if (!running) {
            return;
        }
        if (status != tensorflow::Status::OK()) {
            spdlog::error(MGMT + "hd send {} error '{}'", sendName, status.error_message());
            throw runtime_error("hd send error");
        }
        if (batchId != -1 && resendTime != 0) {
            spdlog::warn(MGMT + "hd send {} batch: {} failed, retry: {} ", sendName, batchId, resendTime);
        }
        resendTime++;
    } while (isNeedResend);
#endif
}

vector<tensorflow::Tensor> HDTransfer::Recv(TransferChannel channel, int channelId, const string& embName)
{
    EASY_FUNCTION()
#ifndef GTEST
    std::vector<tensorflow::Tensor> tensors;
    string recvName = fmt::format("{}_{}_{}", embName, TransferChannel2Str(channel), channelId);
    spdlog::debug("hd transfer try recv:{}", recvName);

    tensorflow::Status status = tensorflow::RecvTensorByAcl(transferChannels[recvName], tensors);
    if (!running) {
        return {};
    }
    if (status != tensorflow::Status::OK()) {
        spdlog::error(MGMT + "{} hd recv error '{}'", recvName, status.error_message());
        throw runtime_error("hd recv error");
    }

    vector<size_t> sizes;
    for (auto& t: tensors) {
        sizes.push_back(t.NumElements());
    }
    spdlog::info("hd transfer recv:{}, size:{}", recvName, sizes);
    return tensors;
#endif
    return {};
}

tuple<acltdtDataset*, size_t> HDTransfer::RecvAcl(TransferChannel channel, int channelId, const string& embName)
{
    EASY_FUNCTION()
#ifndef GTEST
    std::vector<tensorflow::Tensor> tensors;
    string recvName = fmt::format("{}_{}_{}", embName, TransferChannel2Str(channel), channelId);
    spdlog::debug("hd transfer try recv:{}", recvName);
    acltdtDataset* aclDataset = acltdtCreateDataset();
    if (aclDataset == nullptr) {
        throw runtime_error(fmt::format("Failed recv:{}.", recvName).c_str());
    }
    auto aclStatus = acltdtReceiveTensor(transferChannels[recvName], aclDataset, -1 /* no timeout */);
    if (!running) {
        return {nullptr, 0};
    }
    if (aclStatus != ACL_ERROR_NONE && aclStatus != ACL_ERROR_RT_QUEUE_EMPTY) {
        throw runtime_error(fmt::format("Failed receive data from acl channel, acl status:{}", aclStatus).c_str());
    }
    spdlog::info("hd transfer recv:{}", recvName);
    return {aclDataset, acltdtGetDatasetSize(aclDataset)};
#endif
    return {nullptr, 0};
}

size_t HDTransfer::QueryChannelSize(const string& channelName)
{
    size_t size = -1;
#ifndef GTEST
    acltdtQueryChannelSize(transferChannels[channelName], &size);
#endif
    return size;
}
