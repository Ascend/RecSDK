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
int HDTransfer::Init(const vector<EmbInfo>& embInfos, uint32_t localRankId, bool isIncrementalCkpt)
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
    for (const auto& embInfo : embInfos) {
        auto embName = embInfo.name;
        for (int i = 0; i < MAX_CHANNEL_NUM; ++i) {
            CreateChannel(localRankId, embInfo.name, i);
            if (isIncrementalCkpt) {
                CreateChannelForIncrementalCkpt(localRankId, embInfo.name, i);
            }
        }
        // 创建acltdtDataset类型的数据，对等一个Vector<tensor>。同步接口。
        for (int j = 0; j < EMBEDDING_THREAD_NUM; j++) {
            acltdtDataset* dataset = acltdtCreateDataset();
            if (dataset == nullptr) {
                LOG_ERROR("create acltdtDataset failed, table:{}, threadId:{}", embName, j);
                throw runtime_error("create acltdtDataset failed");
            }
            aclDatasets[embInfo.name][j] = dataset;
        }
        if (isIncrementalCkpt) {
            acltdtDataset* dataset = acltdtCreateDataset();
            if (dataset == nullptr) {
                LOG_ERROR("Create acltdtDataset failed, table:{}.", embName);
                throw runtime_error("create acltdtDataset failed");
            }
            aclDatasetsForIncrementalCkpt[embInfo.name] = dataset;
        }
    }
    for (int i = 0; i < MAX_CHANNEL_NUM; ++i) {
        usedChannelsNames[i];
    }
    running = true;
    LOG_INFO(MGMT + "hd_transfer init end");
#endif
    return true;
}

/// 删除所有通道和TDT dataset
void HDTransfer::Destroy()
{
#ifndef GTEST
    running = false;
    LOG_INFO(HD + "destroy channel start");
    for (auto& c : transferChannels) {
        LOG_INFO(HD + "start destroy channel:{}", c.first);
        if (acltdtStopChannel(c.second) != ACL_ERROR_NONE || acltdtDestroyChannel(c.second) != ACL_ERROR_NONE) {
            throw runtime_error("Acl destroy channel failed.");
        }
        LOG_INFO(HD + "destroy channel:{}", c.first);
    }
    for (auto& datasetMap : aclDatasets) {
        for (auto& d : datasetMap.second) {
            if (acltdtDestroyDataset(d.second) != ACL_ERROR_NONE) {
                throw runtime_error("Acl destroy tensor dataset failed.");
            }
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
    for (int c = static_cast<int>(TransferChannel::D2H); c != static_cast<int>(TransferChannel::KEY_D2H); c++) {
        if ((c == static_cast<int>(TransferChannel::SAVE_D2H) || c == static_cast<int>(TransferChannel::SAVE_H2D)) &&
            channelNum == EVAL_CHANNEL_ID) {
            continue;
        }

        auto channel = static_cast<TransferChannel>(c);
        std::string sendName =
            StringFormat("%s_%s_%d", embName.c_str(), TransferChannel2Str(channel).c_str(), channelNum);

        if (TransferChannel2Str(channel) == "all2all" || TransferChannel2Str(channel) == "restore" ||
            TransferChannel2Str(channel) == "lookup" || TransferChannel2Str(channel) == "restore_second" ||
            TransferChannel2Str(channel) == "uniquekeys" || TransferChannel2Str(channel) == "evict" ||
            TransferChannel2Str(channel) == "swap") {
            transferChannels[sendName] = TDT_CREATE_CHANNEL(localRankId, sendName.c_str(), channelSize);
        } else {
            transferChannels[sendName] = TDT_CREATE_CHANNEL(localRankId, sendName.c_str(), PING_PONG_SIZE);
        }
        LOG_INFO("create channel:{} {}", sendName, static_cast<void*>(transferChannels[sendName]));
    }
#endif
}

void HDTransfer::CreateChannelForIncrementalCkpt(const uint32_t localRankId, const string& embName,
                                                 const int channelNum)
{
    int channelSize = GlobalEnv::hdChannelSize;
    LOG_INFO("User config send timestamp and global step channel size:{}.", channelSize);
    int c = static_cast<int>(TransferChannel::KEY_D2H);
    auto channel = static_cast<TransferChannel>(c);
    std::string sendName = StringFormat("%s_%s_%d", embName.c_str(), TransferChannel2Str(channel).c_str(),
                                        channelNum);
    transferChannels[sendName] = TDT_CREATE_CHANNEL(localRankId, sendName.c_str(), PING_PONG_SIZE);
    LOG_INFO("Create channel:{} {}.", sendName, static_cast<void*>(transferChannels[sendName]));
}

/// 将tensor发送到channel
/// \param channel 通道实例
/// \param tensors 待发送数据
/// \param channelId 通道索引（训练/推理）
/// \param embName 表名
/// \param batchId 已处理的batch数
void HDTransfer::Send(TransferChannel channel, const vector<Tensor>& tensors, int channelId, const string& embName,
                      int batchId)
{
    if (!running) {
        return;
    }
#ifndef GTEST
    vector<size_t> sizes;
    for (auto& t : tensors) {
        sizes.push_back(t.NumElements());
    }

    string sendBatchIdType;
    if (channel == TransferChannel::D2H || channel == TransferChannel::H2D) {
        sendBatchIdType = "accumulate";
    }

    string sendName = StringFormat("%s_%s_%d", embName.c_str(), TransferChannel2Str(channel).c_str(), channelId);

    LOG_INFO(HD + "hd transfer send:{}, {} batchId:{}, send count:{}, size list:{}", sendName, sendBatchIdType, batchId,
             sizes.size(), VectorToString(sizes));

    if (sizes.size() == 0) {
        LOG_WARN("tensors num can not be zero");
        return;
    }
    bool isNeedResend = false;
    int resendTime = 0;
    tensorflow::Status status = tensorflow::Status::OK();
    do {
        status =
            tensorflow::SendTensorsByAcl(transferChannels[sendName], ACL_TENSOR_DATA_TENSOR, tensors, isNeedResend);
        if (!running) {
            return;
        }
        if (status != tensorflow::Status::OK()) {
            LOG_ERROR(MGMT + "hd send {} error '{}'", sendName, status.error_message());
            throw runtime_error("hd send error");
        }
        if (batchId != -1 && resendTime != 0) {
            LOG_WARN(MGMT + "hd send: {}, {} batchId: {} failed, retry: {} ", sendName, sendBatchIdType, batchId,
                     resendTime);
        }
        resendTime++;
    } while (isNeedResend);

    if (channel != TransferChannel::EVICT) {
        // Records used channel name in training and used to send EOS later.
        RecordTrainingChannelStr(channel, channelId);
    }
    LOG_DEBUG(HD + "hd transfer send end:{}, {} batchId:{}.", sendName, sendBatchIdType, batchId);
#endif
}

/// 接收从device发送过来的数据（D2H）；使用tfa封装的接口
/// \param channel 通道实例
/// \param channelId 通道索引（训练/推理）
/// \param embName 表名
/// \return
vector<tensorflow::Tensor> HDTransfer::Recv(TransferChannel channel, int channelId, const string& embName)
{
    vector<tensorflow::Tensor> tensors;
#ifndef GTEST
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
    for (auto& t : tensors) {
        sizes.push_back(t.NumElements());
    }
    LOG_INFO("hd transfer recv:{}, size:{} cost:{}ms", recvName, VectorToString(sizes), tc.ElapsedMS());
#endif
    return tensors;
}

/// 接收从device发送过来的数据（D2H）；使用原生的aclTDT接口
/// \param channel 通道实例
/// \param channelId 通道索引（训练/推理）
/// \param embName 表名
/// \return
size_t HDTransfer::RecvAcl(TransferChannel channel, int channelId, const string& embName, int embeddingThreadId,
                           int batchId)
{
    size_t ret = 0;
#ifndef GTEST
    string recvBatchIdType;
    if (channel == TransferChannel::D2H || channel == TransferChannel::H2D) {
        recvBatchIdType = "accumulate";
    }
    string recvName = StringFormat("%s_%s_%d", embName.c_str(), TransferChannel2Str(channel).c_str(), channelId);
    LOG_DEBUG("hd transfer try recv:{}, {} batchId:{}", recvName, recvBatchIdType, batchId);
    TimeCost tc = TimeCost();
    if (aclDatasets[embName][embeddingThreadId] == nullptr) {
        throw runtime_error(StringFormat("Failed recv:%s.", recvName.c_str()).c_str());
    }
    auto aclStatus =
        acltdtReceiveTensor(transferChannels[recvName], aclDatasets[embName][embeddingThreadId], GlobalEnv::aclTimeout);
    if (!running) {
        return 0;
    }
    if (aclStatus != ACL_ERROR_NONE && aclStatus != ACL_ERROR_RT_QUEUE_EMPTY) {
        throw runtime_error(StringFormat("Failed receive data from acl channel, acl status:%d", aclStatus).c_str());
    }
    LOG_INFO("hd transfer recv:{}, {} batchId:{}, cost:{}ms", recvName, recvBatchIdType, batchId, tc.ElapsedMS());
    ret = acltdtGetDatasetSize(aclDatasets[embName][embeddingThreadId]);
#endif
    return ret;
}

size_t HDTransfer::RecvOffsetsAcl(TransferChannel channel, int channelId, const string& embName)
{
    size_t ret = 0;
    string recvName = StringFormat("%s_%s_%d", embName.c_str(), TransferChannel2Str(channel).c_str(), channelId);
    LOG_DEBUG("hd transfer try recv:{}", recvName);
    TimeCost tc = TimeCost();
    if (aclDatasetsForIncrementalCkpt[embName] == nullptr) {
        throw runtime_error(StringFormat("Failed recv:%s.", recvName.c_str()).c_str());
    }
    auto aclStatus = acltdtReceiveTensor(transferChannels[recvName],
                                         aclDatasetsForIncrementalCkpt[embName], GlobalEnv::aclTimeout);
    if (!running) {
        return 0;
    }
    if (aclStatus != ACL_ERROR_NONE && aclStatus != ACL_ERROR_RT_QUEUE_EMPTY) {
        throw runtime_error(StringFormat("Failed receive data from acl channel, acl status:%d", aclStatus).c_str());
    }
    LOG_INFO("hd transfer recv:{}, cost:{}ms", recvName, tc.ElapsedMS());
    ret = acltdtGetDatasetSize(aclDatasetsForIncrementalCkpt[embName]);
    return ret;
}

std::unordered_map<std::string, acltdtChannelHandle*> HDTransfer::GetTransChannel()
{
    return transferChannels;
}

std::unordered_map<int, std::set<std::string>> HDTransfer::GetUsedTransChannel()
{
    return usedChannelsNames;
}

void HDTransfer::ClearTransChannel(int channelId)
{
    LOG_INFO("[CLEAR] Start to clear channel: {}", channelId);

    acltdtDataset* trashDataset = acltdtCreateDataset();
    std::unordered_map<std::string, acltdtChannelHandle*> transChannels = this->GetTransChannel();

    for (auto it = transChannels.begin(); it != transChannels.end(); it++) {
        std::string channelName = it->first;
        auto channelHandle = it->second;

        int table_channel = channelName.back() - '0';
        if (table_channel != channelId) {
            continue;
        }

        size_t initSize = 0;
        acltdtQueryChannelSize(channelHandle, &initSize);
        if (initSize == 0) {
            continue;
        }

        size_t currSize = initSize;
        do {
            auto status = acltdtReceiveTensor(channelHandle, trashDataset, -1);
            if (status != ACL_SUCCESS) {
                LOG_INFO("[CLEAR] Failed to recv eos from channel: {}, error: {}.", channelName, status);
            }
            acltdtQueryChannelSize(channelHandle, &currSize);
        } while (currSize > 0);
        LOG_INFO("[CLEAR] ChannelName: {}, ChannelSize: {} -> {}", channelName, initSize, currSize);
    }

    acltdtDestroyDataset(trashDataset);
}

void HDTransfer::RecordTrainingChannelStr(TransferChannel channel, const int channelId)
{
    std::string channelStr = TransferChannel2Str(channel);
    if (usedChannelsNames[channelId].find(channelStr) != usedChannelsNames[channelId].end()) {
        return;
    }
    std::unique_lock<std::mutex> lock(recordChannelMtx);
    if (usedChannelsNames[channelId].find(channelStr) == usedChannelsNames[channelId].end()) {
        usedChannelsNames[channelId].insert(channelStr);
    }
}
