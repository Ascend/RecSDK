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
int HDTransfer::Init(const vector<EmbInfo>& embInfos, uint32_t localRankId, bool isIncrementalCkpt, bool useLccl)
{
#ifndef GTEST
    LOG_INFO("start init HDTransfer.");
    LOG_INFO("Start aclInit, rank:{}.", localRankId);
    if (!useLccl && !GlobalEnv::useShmSwap) {
        aclError retOk = aclInit(nullptr);
        LOG_INFO("End aclInit, rank:{}.", localRankId);
        if (retOk != ACL_SUCCESS) {
            LOG_ERROR("aclInit failed, rank:{}, errno:{}.", localRankId, retOk);
            return false;
        }
    }
    LOG_INFO("Start aclrtSetDevice, rank:{}.", localRankId);
    auto ret = aclrtSetDevice(static_cast<int32_t>(localRankId));
    if (ret != ACL_ERROR_NONE) {
        LOG_ERROR("aclrtSetDevice failed, rank:{}, error:{}.", localRankId, ret);
        return false;
    }
    LOG_INFO("End aclrtSetDevice, rank:{}.", localRankId);
    for (const auto& embInfo : embInfos) {
        auto embName = embInfo.name;
        for (int i = 0; i < MAX_CHANNEL_NUM; ++i) {
            CreateChannel(localRankId, embInfo.name, i);
            if (isIncrementalCkpt) {
                CreateChannelForIncrementalCkpt(localRankId, embInfo.name, i);
            }
        }
        for (int j = 0; j < EMBEDDING_THREAD_NUM; j++) {
            acltdtDataset* dataset = acltdtCreateDataset();
            if (dataset == nullptr) {
                LOG_ERROR("Create acltdtDataset failed, table:{}, threadId:{}.", embName, j);
                throw runtime_error("Create acltdtDataset failed.");
            }
            aclDatasets[embInfo.name][j] = dataset;
        }
        if (isIncrementalCkpt) {
            acltdtDataset* dataset = acltdtCreateDataset();
            if (dataset == nullptr) {
                LOG_ERROR("Create acltdtDataset failed, table:{}.", embName);
                throw runtime_error("Create acltdtDataset failed.");
            }
            aclDatasetsForIncrementalCkpt[embInfo.name] = dataset;
        }
    }
    for (int i = 0; i < MAX_CHANNEL_NUM; ++i) {
        usedChannelsNames[i];
    }
    running = true;
    localDeviceId = localRankId;
    LOG_INFO("End init HDTransfer.");
#endif
    return true;
}

/// 删除所有通道和TDT dataset
void HDTransfer::Destroy()
{
#ifndef GTEST
    running = false;
    LOG_INFO("Start destroy channel.");
    for (auto& c : transferChannels) {
        LOG_INFO("Start destroy channel:{}.", c.first);
        if (acltdtStopChannel(c.second) != ACL_ERROR_NONE || acltdtDestroyChannel(c.second) != ACL_ERROR_NONE) {
            throw runtime_error("Acl destroy channel failed.");
        }
        LOG_INFO("End destroy channel:{}.", c.first);
    }
    for (auto& datasetMap : aclDatasets) {
        for (auto& d : datasetMap.second) {
            if (acltdtDestroyDataset(d.second) != ACL_ERROR_NONE) {
                throw runtime_error("Acl destroy tensor dataset failed.");
            }
        }
    }
    if (GlobalEnv::useShmSwap) {
        FreeShmAddr(localDeviceId);
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
    LOG_INFO("Start create channel, size:{}.", channelSize);
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
            TransferChannel2Str(channel) == "swap" || TransferChannel2Str(channel) == "mask" ||
            TransferChannel2Str(channel) == "recvshape") {
            transferChannels[sendName] = TDT_CREATE_CHANNEL(localRankId, sendName.c_str(), channelSize);
        } else {
            transferChannels[sendName] = TDT_CREATE_CHANNEL(localRankId, sendName.c_str(), PING_PONG_SIZE);
        }
        LOG_INFO("Create channel:{}", sendName);
    }
    LOG_INFO("End create channel.");
#endif
}

void HDTransfer::CreateChannelForIncrementalCkpt(const uint32_t localRankId, const string& embName,
                                                 const int channelNum)
{
#ifndef GTEST
    int channelSize = GlobalEnv::hdChannelSize;
    LOG_INFO("Start create channel for IncrementalCkpt, size:{}.", channelSize);
    int c = static_cast<int>(TransferChannel::KEY_D2H);
    auto channel = static_cast<TransferChannel>(c);
    std::string sendName = StringFormat("%s_%s_%d", embName.c_str(), TransferChannel2Str(channel).c_str(), channelNum);
    transferChannels[sendName] = TDT_CREATE_CHANNEL(localRankId, sendName.c_str(), PING_PONG_SIZE);
    LOG_INFO("Create channel:{}.", sendName);
#endif
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

    if (sizes.size() == 0) {
        LOG_WARN("No elements to send, channelName:{}, sendBatchIdType:{}, batchId:{}.", sendName, sendBatchIdType,
                 batchId);
        return;
    }

    LOG_INFO("Start sending, channelName:{}, sendBatchIdType:{}, batchId:{}, shape:{}.", sendName, sendBatchIdType,
             batchId, VectorToString(sizes));
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
            LOG_ERROR("Send tensor failed, channelName:{}, sendBatchIdType:{}, batchId:{}, error:{}.", sendName,
                      sendBatchIdType, batchId, status.error_message());
            throw runtime_error("Send tensor failed");
        }
        if (batchId != -1 && resendTime != 0) {
            LOG_WARN("Try resend, channelName:{}, sendBatchIdType:{}, batchId:{}, retry:{}.", sendName, sendBatchIdType,
                     batchId, resendTime);
        }
        resendTime++;
    } while (isNeedResend);

    if (channel != TransferChannel::EVICT) {
        // Records used channel name in training and used to send EOS later.
        RecordTrainingChannelStr(channel, channelId);
    }
    LOG_DEBUG("End sending, channelName:{}, sendBatchIdType:{}, batchId:{}.", sendName, sendBatchIdType, batchId);
#endif
}

size_t HDTransfer::RecvByShm(RmaShmHeader* queueHeader, float*& ptr, int64_t& dim0, bool& emptyFlag)
{
    if ((queueHeader->seqIn - queueHeader->seqOut) == 0) {
        emptyFlag = true;
        return 0;
    }

    RmaShmData *dataHead = ShmDequeuePre(queueHeader);
    if (dataHead != nullptr) {
        LOG_DEBUG("Shm recv data-seq: {}, total-len: {}, dim-num: {}, dim-0: {}, dim-1: {}.",
                  dataHead->sequence, dataHead->totalLen, dataHead->dimNum, dataHead->dims[0], dataHead->dims[1]);

        ptr = reinterpret_cast<float*>(GetDataAddr(dataHead));
        dim0 = dataHead->dims[0];
        emptyFlag = false;
        return dataHead->dataLen;
    }
    emptyFlag = true;
    return 0;
}

size_t HDTransfer::RecvMteShm(string& name, float*& ptr, int64_t& dim0, int batchId)
{
    size_t ret = 0;
    string recvName = StringFormat("%s_%d", name, localDeviceId);
    LOG_DEBUG("Start receive, channelName:{}, batchId:{}.", recvName, batchId);
    TimeCost tc = TimeCost();

    auto *shmAddr = GetHostAddr(recvName);
    if (shmAddr == nullptr) {
        auto error = Error(ModuleName::M_HD_TRANSFER, ErrorType::INVALID_ARGUMENT,
                           StringFormat("Failed to find valid shm for channel: %s device: %d.",
                                        recvName.c_str(), localDeviceId));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString());
    }

    do {
        bool emptyFlag = false;
        ret = RecvByShm(reinterpret_cast<RmaShmHeader *>(shmAddr), ptr, dim0, emptyFlag);
        if (!emptyFlag) {
            break;
        }

        if (!running) {
            return 0;
        }
    } while (running);

    LOG_INFO("End receive, channelName:{}, batchId:{}, cost:{}ms.", recvName, batchId, tc.ElapsedMS());
    return ret;
}

void HDTransfer::DequeueShm(TransferChannel channel, int channelId, const string& embName)
{
    string recvName = StringFormat("%s_%s_%d_%d", embName.c_str(), TransferChannel2Str(channel).c_str(),
                                   channelId, localDeviceId);
    RmaShmHeader *queueHeader = reinterpret_cast<RmaShmHeader *>(GetHostAddr(recvName));
    if (queueHeader == nullptr) {
        auto error = Error(ModuleName::M_HD_TRANSFER, ErrorType::INVALID_ARGUMENT,
                           StringFormat("Failed to find valid shm for channel: %s device: %d.",
                                        recvName.c_str(), localDeviceId));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString());
    }
    if (ShmDequeue(queueHeader) == nullptr) {
        LOG_DEBUG("Shm queue {} is already empty.", recvName);
    }
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
    LOG_DEBUG("Start receive, channelName:{}, recvBatchIdType:{}, batchId:{}.", recvName, recvBatchIdType, batchId);
    TimeCost tc = TimeCost();
    if (aclDatasets[embName][embeddingThreadId] == nullptr) {
        throw runtime_error(StringFormat("Find aclDatasets not init, should call HDTransfer:Init first, channelName:%s",
                                         recvName.c_str())
                                .c_str());
    }
    auto aclStatus =
        acltdtReceiveTensor(transferChannels[recvName], aclDatasets[embName][embeddingThreadId], GlobalEnv::aclTimeout);
    if (!running) {
        return 0;
    }
    if (aclStatus != ACL_ERROR_NONE && aclStatus != ACL_ERROR_RT_QUEUE_EMPTY) {
        throw runtime_error(StringFormat("Failed receive data from acl channel, acl status:%d.", aclStatus).c_str());
    }
    LOG_INFO("End receive, channelName:{}, recvBatchIdType{}, batchId:{}, cost:{}ms.", recvName, recvBatchIdType,
             batchId, tc.ElapsedMS());
    ret = acltdtGetDatasetSize(aclDatasets[embName][embeddingThreadId]);
#endif
    return ret;
}

size_t HDTransfer::RecvOffsetsAcl(TransferChannel channel, int channelId, const string& embName)
{
    size_t ret = 0;
#ifndef GTEST
    string recvName = StringFormat("%s_%s_%d", embName.c_str(), TransferChannel2Str(channel).c_str(), channelId);
    LOG_DEBUG("Start receive, channelName:{}.", recvName);
    TimeCost tc = TimeCost();
    if (aclDatasetsForIncrementalCkpt[embName] == nullptr) {
        throw runtime_error(StringFormat("Failed recv:%s.", recvName.c_str()).c_str());
    }
    auto aclStatus =
        acltdtReceiveTensor(transferChannels[recvName], aclDatasetsForIncrementalCkpt[embName], GlobalEnv::aclTimeout);
    if (!running) {
        return 0;
    }
    if (aclStatus != ACL_ERROR_NONE && aclStatus != ACL_ERROR_RT_QUEUE_EMPTY) {
        throw runtime_error(StringFormat("Failed receive data from acl channel, acl status:%d", aclStatus).c_str());
    }
    LOG_INFO("End receive, channelName:{}, cost:{}ms.", recvName, tc.ElapsedMS());
    ret = acltdtGetDatasetSize(aclDatasetsForIncrementalCkpt[embName]);
#endif
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
    LOG_INFO("Start to clear channel, channelId:{}", channelId);
#ifndef GTEST
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
                LOG_INFO("Failed to recv eos from channelName:{}, error:{}.", channelName, status);
            }
            acltdtQueryChannelSize(channelHandle, &currSize);
        } while (currSize > 0);
        LOG_INFO("Clear channel, ChannelName:{}, ChannelSize before:{}, after:{}", channelName, initSize, currSize);
    }

    acltdtDestroyDataset(trashDataset);
    if (GlobalEnv::useShmSwap) {
        ClearShmQueue();
    }
#endif
}

void HDTransfer::RecordTrainingChannelStr(TransferChannel channel, const int channelId)
{
#ifndef GTEST
    std::string channelStr = TransferChannel2Str(channel);
    if (usedChannelsNames[channelId].find(channelStr) != usedChannelsNames[channelId].end()) {
        return;
    }
    std::unique_lock<std::mutex> lock(recordChannelMtx);
    if (usedChannelsNames[channelId].find(channelStr) == usedChannelsNames[channelId].end()) {
        usedChannelsNames[channelId].insert(channelStr);
    }
#endif
}
