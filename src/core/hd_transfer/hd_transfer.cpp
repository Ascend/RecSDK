/*
* Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
* Description: common module
* Author: MindX SDK
* Date: 2022/11/15
*/
#include "hd_transfer.h"
#include <fstream>
#include "utils/common.h"
#include "utils/time_cost.h"

using namespace MxRec;
using namespace std;

/// 1. acl初始化 2. 设置device 3. 为每张表创建数据传输通道
/// \param embInfos 稀疏表元信息类的list
/// \param localRankId 设备逻辑ID
/// \return
int HDTransfer::Init(const vector<EmbInfo>& embInfos, uint32_t localRankId)
{
#ifndef GTEST
    LOG_INFO(MGMT + "begin hd_transfer initialize, rank:{}", localRankId);
    // 使用AscendCL接口开发应用时，必须先调用aclInit接口，否则可能会导致后续系统内部资源初始化出错，进而导致其它业务异常。
    aclError retOk = aclInit(nullptr);
    LOG_INFO(MGMT + "end aclInit, rank:{}", localRankId);
    if (retOk != ACL_SUCCESS) {
        LOG_ERROR(MGMT + "aclInit fail, rank:{}, errno:{}", localRankId, retOk);
        return false;
    }
    LOG_INFO(MGMT + "start Set device, rank:{}", localRankId);
    // 指定当前进程或线程中用于运算的Device，同时隐式创建默认Context
    auto ret = aclrtSetDevice(static_cast<int32_t>(localRankId));
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("Set device failed, device_id:{}", localRankId);
        return false;
    }
    LOG_INFO(MGMT + "end Set device, rank:{}", localRankId);
    for (const auto& embInfo: embInfos) {
        auto embName = embInfo.name;
        for (int i = 0; i < MAX_CHANNEL_NUM; ++i) {
            CreateChannel(localRankId, embInfo.name, i);
        }
        // 创建acltdtDataset类型的数据，对等一个Vector<tensor>。同步接口。
        aclDatasets[embInfo.name] = acltdtCreateDataset();
    }
    running = true;
    LOG(INFO) << "hd_transfer init";
#endif
    return true;
}

/// 删除所有通道和TDT dataset
void HDTransfer::Destroy()
{
#ifndef GTEST
    running = false;
    LOG_INFO(HD + "destroy channel start");
    for (auto& c: transferChannels) {
        LOG_INFO(HD + "start destroy channel:{}", c.first);
        if (acltdtStopChannel(c.second) != ACL_ERROR_NONE || acltdtDestroyChannel(c.second) != ACL_ERROR_NONE) {
            throw runtime_error("Acl destroy channel failed.");
        }
        LOG_INFO(HD + "destroy channel:{}", c.first);
    }
    for (auto& d: aclDatasets) {
        if (acltdtDestroyDataset(d.second) != ACL_ERROR_NONE) {
            throw runtime_error("Acl destroy tensor dataset failed.");
        }
    }
    aclFinalize();
#endif
}

/// 为每张表创建相应的数据传输通道（all2ll、restore、lookup等）
/// \param localRankId 设备逻辑ID
/// \param embName 表名
/// \param channelNum 通道索引
void HDTransfer::CreateChannel(const uint32_t localRankId, const string& embName, const int channelNum)
{
#ifndef GTEST
    int channelSize = GlobalEnv::hdChannelSize;
    LOG_INFO("user config all2all restore lookup channel size:{}", channelSize);
    for (int c = static_cast<int>(TransferChannel::D2H); c != static_cast<int>(TransferChannel::INVALID); c++) {
        auto channel = static_cast<TransferChannel>(c);
        string sendName = StringFormat(
            "%s_%s_%d", embName.c_str(), TransferChannel2Str(channel).c_str(), channelNum
        );
        if (TransferChannel2Str(channel) == "all2all" ||
            TransferChannel2Str(channel) == "restore" ||
            TransferChannel2Str(channel) == "lookup"  ||
            TransferChannel2Str(channel) == "restore_second" ||
            TransferChannel2Str(channel) == "uniquekeys" ||
            TransferChannel2Str(channel) == "evict"  /* for noDDR */
                ) {
            transferChannels[sendName] = tdtCreateChannel(localRankId, sendName.c_str(), channelSize);
        } else {
            transferChannels[sendName] = tdtCreateChannel(localRankId, sendName.c_str(), PING_PONG_SIZE);
        }
        LOG_INFO("create channel:{} {}", sendName, static_cast<void*>(transferChannels[sendName]));
    }
#endif
}

/// 将tensor发送到channel
/// \param channel 通道实例
/// \param tensors 待发送数据
/// \param channelId 通道索引（训练/推理）
/// \param embName 表名
/// \param batchId 已处理的batch数
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
    string sendName = StringFormat("%s_%s_%d", embName.c_str(), TransferChannel2Str(channel).c_str(), channelId);

    LOG_INFO(HD + "hd transfer send {}, send count is {}, size list:{}",
             sendName, sizes.size(), VectorToString(sizes));

    if (sizes.size() == 0) {
        LOG_WARN("tensors num can not be zero");
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
            LOG_ERROR(MGMT + "hd send {} error '{}'", sendName, status.error_message());
            throw runtime_error("hd send error");
        }
        if (batchId != -1 && resendTime != 0) {
            LOG_WARN(MGMT + "hd send {} batch: {} failed, retry: {} ", sendName, batchId, resendTime);
        }
        resendTime++;
    } while (isNeedResend);
#endif
}

/// 接收从device发送过来的数据（D2H）；使用tfa封装的接口
/// \param channel 通道实例
/// \param channelId 通道索引（训练/推理）
/// \param embName 表名
/// \return
vector<tensorflow::Tensor> HDTransfer::Recv(TransferChannel channel, int channelId, const string& embName)
{
    EASY_FUNCTION()
#ifndef GTEST
    std::vector<tensorflow::Tensor> tensors;
    string recvName = StringFormat("%s_%s_%d", embName.c_str(), TransferChannel2Str(channel).c_str(), channelId);
    LOG_DEBUG("hd transfer try recv:{}", recvName);
    TimeCost tc = TimeCost();
    tensorflow::Status status = tensorflow::RecvTensorByAcl(transferChannels[recvName], tensors);
    if (!running) {
        return {};
    }
    if (status != tensorflow::Status::OK()) {
        LOG_ERROR(MGMT + "{} hd recv error '{}'", recvName, status.error_message());
        throw runtime_error("hd recv error");
    }

    vector<size_t> sizes;
    for (auto& t: tensors) {
        sizes.push_back(t.NumElements());
    }
    LOG_INFO("hd transfer recv:{}, size:{} cost:{}ms", recvName, VectorToString(sizes), tc.ElapsedMS());
    return tensors;
#endif
}

/// 接收从device发送过来的数据（D2H）, updateEmbV2函数使用；使用原生的aclTDT接口
/// \param channel 通道实例
/// \param channelId 通道索引（训练/推理）
/// \param embName 表名
/// \return
size_t HDTransfer::RecvAcl(TransferChannel channel, int channelId, const string& embName)
{
    EASY_FUNCTION()
#ifndef GTEST
    std::vector<tensorflow::Tensor> tensors;
    string recvName = StringFormat("%s_%s_%d", embName.c_str(), TransferChannel2Str(channel).c_str(), channelId);
    LOG_DEBUG("hd transfer try recv:{}", recvName);
    TimeCost tc = TimeCost();
    if (aclDatasets[embName] == nullptr) {
        throw runtime_error(StringFormat("Failed recv:%s.", recvName.c_str()).c_str());
    }
    auto aclStatus = acltdtReceiveTensor(transferChannels[recvName], aclDatasets[embName], GlobalEnv::aclTimeout);
    if (!running) {
        return 0;
    }
    if (aclStatus != ACL_ERROR_NONE && aclStatus != ACL_ERROR_RT_QUEUE_EMPTY) {
        throw runtime_error(StringFormat("Failed receive data from acl channel, acl status:%d", aclStatus).c_str());
    }
    LOG_INFO("hd transfer recv:{} cost:{}ms", recvName, tc.ElapsedMS());
    return acltdtGetDatasetSize(aclDatasets[embName]);
#endif
}

std::unordered_map<std::string, acltdtChannelHandle*> HDTransfer::GetTransChannel()
{
    return transferChannels;
}