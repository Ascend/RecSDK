/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: hybrid mgmt module,Record the number of program running steps,
 * manage blocking and wakeup
 * Author: MindX SDK
 * Date: 2023/08/15
 */
#include <thread>

#include "utils/common.h"
#include "hybrid_mgmt_block.h"

/// 检查当前hybrid是否运行到了应该阻塞的位置
/// \param channelId train 0 eval 1
void HybridMgmtBlock::CheckAndSetBlock(int channelId)
{
    if (stepsInterval[channelId] == -1) {
        return;
    }
    if (stepsInterval[channelId] == 0) {
        // 为0应该阻塞，并且避免下面除0的逻辑
        isBlock[channelId] = true;
        return;
    }
    if (hybridBatchId[channelId] % stepsInterval[channelId] == 0) {
        isBlock[channelId] = true;
    }
}

/// 检查当前是否进行了数据通道切换，如果进行了数据通道切换则进行参数校验
/// 通过python侧的batchId和hybrid的batchId当前的步数是否到了唤醒阻塞线程的步数
/// \param channelId train 0 eval 1
void HybridMgmtBlock::CheckAndNotifyWake(int channelId)
{
    CheckValid(channelId);
    if (pythonBatchId[channelId] >= hybridBatchId[channelId]) {
        isBlock[channelId] = false;
    }
}

/// 如果检查参数不合理，涉及到抛出异常，需要先等待，有可能是数据传输未完成。
/// \param channelId train 0 eval 1
bool HybridMgmtBlock::WaitValid(int channelId)
{
    // 等待hybrid处理完成
    int reTryNumber = 100;
    VLOG(INFO) << StringFormat(HYBRID_BLOCKING +
                               "check step invalid, wait", channelId, hybridBatchId[channelId]);
    // 等待hybrid处理完成后再一次唤醒
    while (pythonBatchId[lastRunChannelId] != hybridBatchId[lastRunChannelId] and isRunning) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10ms));
        reTryNumber--;
        if (reTryNumber <= 0) {
            break;
        }
    }

    if (pythonBatchId[channelId] == hybridBatchId[channelId]) {
        return true;
    } else {
        // 如果等待python侧处理较长时间后hybrid依旧无法追赶上python则异常
        return false;
    }
}

void HybridMgmtBlock::CountPythonStep(int channelId)
{
    // 相应的通知计数
    pythonBatchId[channelId]++;
}

/// 检查是否进行了通道切换，检查当前的step是否合理
/// \param channelId
void HybridMgmtBlock::CheckValid(int channelId)
{
    // 通道没有切换，不用处理
    if (lastRunChannelId == channelId) {
        return;
    }
    // 当python侧第一次调用时，此时跳过参数检查
    if (lastRunChannelId == -1) {
        VLOG(GLOG_DEBUG) << StringFormat(HYBRID_BLOCKING +
        "The data channel was called for the first time, and the parameters were "
        "checked to be normal channelId %d hybridBatchId %d", channelId, hybridBatchId[channelId]);

        lastRunChannelId = channelId;
        return;
    }
    // 在通道切换时，hybrid预处理的batch与python的一致。
    if (pythonBatchId[lastRunChannelId] == hybridBatchId[lastRunChannelId]) {
        VLOG(GLOG_DEBUG) << StringFormat(HYBRID_BLOCKING +
        "HybridMgmt is switching data channels and checking for normal parameters. he number of steps "
        "in the previous round is lastRunChannelId %d pythonBatchId %d hybridBatchId %d",
        lastRunChannelId,  pythonBatchId[lastRunChannelId], hybridBatchId[lastRunChannelId]);
    } else if (pythonBatchId[lastRunChannelId] < hybridBatchId[lastRunChannelId]) {
        // 在通道切换时，上一个通道处理的数据超出了python侧的调用
        if (!WaitValid(lastRunChannelId)) {
            throw HybridMgmtBlockingException("when channel switch");
        }
    } else {
        // 在通道切换时，hybrid处理的数据还没有赶上python侧，此时需要等待hybrid处理完成
        VLOG(INFO) << StringFormat(HYBRID_BLOCKING +
        "When switching data channels, it was found that HybridMgmt processed less data than the "
        "Python side.In this case, after reading the dataset, the Python side called it again, but it was "
        "interrupted midway,which did not affect the subsequent calls lastRunChannelId %d hybridBatchId %d",
        lastRunChannelId, hybridBatchId[lastRunChannelId]);
    }
    lastRunChannelId = channelId;
    return;
}

/// 进行阻塞操作
/// \param channelId train 0 eval 1
void HybridMgmtBlock::DoBlock(int channelId)
{
    // 通道没有切换，不用处理
    VLOG(GLOG_DEBUG) << StringFormat(HYBRID_BLOCKING +
    "HybridMgmt starts blocking channelId %d hybridBatchId %d",  channelId, hybridBatchId[channelId]);

    while (isBlock[channelId]) {
        std::this_thread::sleep_for(SLEEP_MS);
        if (!isRunning) {
            return;
        }
    }
    VLOG(GLOG_DEBUG) << StringFormat(HYBRID_BLOCKING +
    "HybridMgmt is starting to wake up channelId %d hybridBatchId %d",  channelId, hybridBatchId[channelId]);
    return;
}

/// 重置所有的步数，主要用于图重构的情况，readembedkey算子重建
/// \param channelId channelId  train 0 eval 1
void HybridMgmtBlock::ResetAll(int channelId)
{
    VLOG(GLOG_DEBUG) << StringFormat(HYBRID_BLOCKING +
    "Hybridmgmt is resetting data channelId %d hybridBatchId %d", channelId, hybridBatchId[channelId]);

    readEmbedBatchId[channelId] = 0;
    pythonBatchId[channelId] = 0;
    hybridBatchId[channelId] = 0;
    isBlock[channelId] = false;
    // eval train通道的sparse 同时进行重置，以防出现sparse id失效的问题
    uniqueSparseLookID[EVAL_CHANNEL_ID] = -1;
    uniqueSparseLookID[TRAIN_CHANNEL_ID] = -1;
}

/// 检查当前的步数是否可以进行save
/// \return 0 is legal, 1 需要回退一步, -1 表示错误
int HybridMgmtBlock::CheckSaveEmbdMapValid()
{
    // 检查数据通道此时的HashMap是否被提前处理了
    if (pythonBatchId[lastRunChannelId] >= hybridBatchId[lastRunChannelId]) {
        VLOG(GLOG_DEBUG) << StringFormat(HYBRID_BLOCKING +
        "HybridMgmt is checking the step and checking that the parameters are normal. "
        "The number of steps in the previous round is "
        "lastRunChannelId %d pythonBatchId %d hybridBatchId %d",
        lastRunChannelId, pythonBatchId[lastRunChannelId], hybridBatchId[lastRunChannelId]);
        return 0;
    } else if (pythonBatchId[lastRunChannelId] + 1 == hybridBatchId[lastRunChannelId]) {
        // 在通道切换时，上一个通道处理的数据超出了python侧的调用
        VLOG(INFO) << StringFormat(HYBRID_BLOCKING +
        "HybridMgmt is checking the step, and the parameters have been processed one step "
        "in advance. The number of steps in the previous round was "
        "lastRunChannelId %d pythonBatchId %d hybridBatchId %d",
        lastRunChannelId, pythonBatchId[lastRunChannelId], hybridBatchId[lastRunChannelId]);

        return 1;
    } else {
        // 在通道切换时，hybrid处理的数据还没有赶上python侧，此时需要等待hybrid处理完成
        VLOG(GLOG_DEBUG) << StringFormat(HYBRID_BLOCKING + "ERROR FLAG lastRunChannelId %d hybridBatchId %d",
                                         lastRunChannelId, hybridBatchId[lastRunChannelId]);
        return -1;
    }
}

bool HybridMgmtBlock::GetBlockStatus(int channelId)
{
    return isBlock[channelId];
}

void HybridMgmtBlock::SetBlockStatus(int channelId, bool block)
{
    isBlock[channelId] = block;
}

/// python侧调用的npu.outfeed_enqueue_op 发送的消息。用来判断当前python执行的步数
void HybridMgmtBlock::StartNotifySignalMonitor()
{
#ifndef GTEST
    auto fn = [this](int channelId) {
        while (isRunning) {
            std::vector<tensorflow::Tensor> tensors;
            tensorflow::Status status = tensorflow::RecvTensorByAcl(aclHandles[channelId], tensors);
            if (!isRunning) {
                break;
            }
            if (status != tensorflow::Status::OK()) {
                LOG(ERROR) << StringFormat(HYBRID_BLOCKING +
                "%s hd recv error '%s'", d2hChannelName[channelId].c_str(), status.error_message().c_str());
                throw runtime_error("rev error");
            }
            VLOG(GLOG_DEBUG) << StringFormat(HYBRID_BLOCKING +
            "send message to hybrid channelId %d pythonBatchId %d hybridBatchId %d",
            channelId, pythonBatchId[channelId], hybridBatchId[channelId]);

            int sparseLookupId = *tensors[0].flat<int32>().data();
            VLOG(GLOG_DEBUG) << StringFormat(HYBRID_BLOCKING +
            "send sparse_lookup_id channel %d sparse id %d unique id %d",
            channelId, sparseLookupId, uniqueSparseLookID[channelId]);

            if (uniqueSparseLookID[channelId] == -1) {
                // 初始化，只有第一个sparse loop id能进行计数和唤
                uniqueSparseLookID[channelId] = sparseLookupId;
            }
            // 只被计数一次
            if (sparseLookupId == uniqueSparseLookID[channelId]) {
                // 只有最先来的id才能进行唤醒和计数
                CheckAndNotifyWake(channelId);
                CountPythonStep(channelId);
            }
        }
        LOG(INFO) << StringFormat(HYBRID_BLOCKING + "BLOCKING thread stop");
    };
    uint32_t localRankId = rankInfo.deviceId;
    for (int channelId = 0; channelId < MAX_CHANNEL_NUM; ++channelId) {
        d2hChannelName[channelId] = StringFormat(D2H_CHANNEL_NAME_PRE + "%d", channelId);
        auto aclChannelHandle = tdtCreateChannel(localRankId, d2hChannelName[channelId].c_str(), PING_PONG_SIZE);
        LOG(INFO) << StringFormat(HYBRID_BLOCKING + " %d %s", localRankId, d2hChannelName[channelId].c_str());
        aclHandles[channelId] = aclChannelHandle;
        procThreads.emplace_back(std::make_unique<std::thread>(fn, channelId));
    }
#endif
}

void  HybridMgmtBlock::Destroy()
{
    if (!isRunning) {
        // 已经销毁过了，不用再次销毁会报错
        return;
    }
    isRunning = false;
#ifndef GTEST
    for (int channelId = 0; channelId < MAX_CHANNEL_NUM; ++channelId) {
        tensorflow::StopRecvTensorByAcl(&aclHandles[channelId], d2hChannelName[channelId]);
        procThreads[channelId]->join();
    }
    LOG(INFO) << StringFormat(HYBRID_BLOCKING + "BLOCKING stop");
#endif
}


void HybridMgmtBlock::SetRankInfo(RankInfo rankInfo)
{
    this->stepsInterval[0] = rankInfo.maxStep[0];
    this->stepsInterval[1] = rankInfo.maxStep[1];
    this->rankInfo = rankInfo;
};

void HybridMgmtBlock::SetStepInterval(int trainStep, int evalStep)
{
    this->stepsInterval[0] = trainStep;
    this->stepsInterval[1] = evalStep;
};

HybridMgmtBlock::~HybridMgmtBlock()
{
    Destroy();
}