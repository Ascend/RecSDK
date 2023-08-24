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
    LOG(INFO) << StringFormat(MGMT + "begin hd_transfer initialize, rank:%d", localRankId);
    // 使用AscendCL接口开发应用时，必须先调用aclInit接口，否则可能会导致后续系统内部资源初始化出错，进而导致其它业务异常。
    aclError retOk = aclInit(nullptr);
    LOG(INFO) << StringFormat(MGMT + "end aclInit, rank:%d", localRankId);
    if (retOk != ACL_SUCCESS) {
        LOG(ERROR) << StringFormat(MGMT + "aclInit fail, rank:%d, errno:%d", localRankId, retOk);
        return false;
    }
    LOG(INFO) << StringFormat(MGMT + "start Set device, rank:%d", localRankId);
    // 指定当前进程或线程中用于运算的Device，同时隐式创建默认Context
    auto ret = aclrtSetDevice(static_cast<int32_t>(localRankId));
    if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << StringFormat("Set device failed, device_id:%d", localRankId);
        return false;
    }
    LOG(INFO) << StringFormat(MGMT + "end Set device, rank:%d", localRankId);
    for (const auto& embInfo: embInfos) {
        auto embName = embInfo.name;
        for (int i = 0; i < MAX_CHANNEL_NUM; ++i) {
            CreateChannel(localRankId, embInfo.name, i);
        }
        // 创建acltdtDataset类型的数据，对等一个Vector<tensor>。同步接口。
        aclDatasets[embInfo.name] = acltdtCreateDataset();
    }
    const int defaultAclTimeout = -1;
    this->timeout = defaultAclTimeout;
    const char *envTimeout = getenv("AclTimeout");
    if (envTimeout != nullptr) {
        try {
            int32_t tmp = std::stoi(envTimeout);
            if (tmp >= -1 && tmp <= INT32_MAX) {
                this->timeout = tmp;
                LOG(INFO) << StringFormat("Succeed to parse ${env:AclTimeout}: %d", tmp);
            } else {
                LOG(ERROR) << StringFormat("Failed to parse ${env:AclTimeout}: %d, expected in (0, INT32_MAX)", tmp);
            }
        } catch (const std::invalid_argument &e) {
            LOG(ERROR) << StringFormat("Failed to parse ${env:AclTimeout}: %s, expected a integer, set to default: %d",
                envTimeout, defaultAclTimeout);
        }
    }
    VLOG(GLOG_DEBUG) << StringFormat("hd transfer timeout:%d", timeout);
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
    LOG(INFO) << (HD + "destroy channel start");
    for (auto& c: transferChannels) {
        LOG(INFO) << StringFormat(HD + "start destroy channel:%s", c.first.c_str());
        tensorflow::StopRecvTensorByAcl(&c.second, c.first);
        LOG(INFO) << StringFormat(HD + "destroy channel:%s", c.first.c_str());
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
    int channelSize;
    const char* env = getenv("HD_CHANNEL_SIZE");
    if (env == nullptr) {
        channelSize = LARGE_CHANNEL_SIZE;
    } else {
        try {
            channelSize = stoi(env);
        } catch (const std::invalid_argument& e) {
            LOG(WARNING) << StringFormat("wrong HD_CHANNEL_SIZE env %s", e.what());
            channelSize = LARGE_CHANNEL_SIZE;
        } catch (const std::out_of_range& e) {
            LOG(WARNING) << StringFormat("wrong HD_CHANNEL_SIZE env %s", e.what());
            channelSize = LARGE_CHANNEL_SIZE;
        }
        if (channelSize <= 0) {
            channelSize = LARGE_CHANNEL_SIZE;
        }
    }
    LOG(INFO) << StringFormat("user config all2all restore lookup channel size:%d", channelSize);
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
        LOG(INFO) << StringFormat(
            "create channel:%s %d", sendName.c_str(), static_cast<void*>(transferChannels[sendName])
        );
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

    if (g_glogLevel >= INFO) {
        LOG(INFO) << StringFormat(
            HD + "hd transfer send %s, send count is %d, size list:%s",
            sendName.c_str(), sizes.size(), VectorToString(sizes).c_str()
        );
    }

    if (sizes.size() == 0) {
        LOG(WARNING) << "tensors num can not be zero";
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
            LOG(ERROR) << StringFormat(
                MGMT + "hd send %s error '%s'", sendName.c_str(), status.error_message().c_str()
            );
            throw runtime_error("hd send error");
        }
        if (batchId != -1 && resendTime != 0) {
            LOG(WARNING) << StringFormat(
                MGMT + "hd send %s batch: %d failed, retry: %d ", sendName.c_str(), batchId, resendTime
            );
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
    VLOG(GLOG_DEBUG) << StringFormat("hd transfer try recv:%s", recvName.c_str());
    TimeCost tc = TimeCost();
    tensorflow::Status status = tensorflow::RecvTensorByAcl(transferChannels[recvName], tensors);
    if (!running) {
        return {};
    }
    if (status != tensorflow::Status::OK()) {
        LOG(ERROR) << StringFormat(MGMT + "%s hd recv error '%s'", recvName.c_str(), status.error_message().c_str());
        throw runtime_error("hd recv error");
    }

    vector<size_t> sizes;
    for (auto& t: tensors) {
        sizes.push_back(t.NumElements());
    }
    if (g_glogLevel >= INFO) {
        LOG(INFO) << StringFormat(
            "hd transfer recv:%s, size:%d cost:%dms", recvName.c_str(), VectorToString(sizes).c_str(), tc.ElapsedMS()
        );
    }
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
    VLOG(GLOG_DEBUG) << StringFormat("hd transfer try recv:%s", recvName.c_str());
    TimeCost tc = TimeCost();
    if (aclDatasets[embName] == nullptr) {
        throw runtime_error(StringFormat("Failed recv:%s.", recvName.c_str()).c_str());
    }
    auto aclStatus = acltdtReceiveTensor(transferChannels[recvName], aclDatasets[embName], timeout /*-1 no timeout */);
    if (!running) {
        return 0;
    }
    if (aclStatus != ACL_ERROR_NONE && aclStatus != ACL_ERROR_RT_QUEUE_EMPTY) {
        throw runtime_error(StringFormat("Failed receive data from acl channel, acl status:%d", aclStatus).c_str());
    }
    LOG(INFO) << StringFormat("hd transfer recv:%s cost:%dms", recvName.c_str(), tc.ElapsedMS());
    return acltdtGetDatasetSize(aclDatasets[embName]);
#endif
}
