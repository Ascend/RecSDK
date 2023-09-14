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
    // 判断save时候的阻塞情况
    // 当在进行训练通道，且save interval不为0和-1（不需要阻塞），且运行到了需要阻塞的步骤
    if (channelId==TRAIN_CHANNEL_ID && saveInterval!=0
        && saveInterval!=-1 && hybridBatchId[TRAIN_CHANNEL_ID]%saveInterval==0) {
        LOG_DEBUG(HYBRID_BLOCKING + "blocking by save saveInterval {} pythonBatchId {} hybridBatchId {}",
                  saveInterval, pythonBatchId[channelId], hybridBatchId[channelId]);
        isBlock[TRAIN_CHANNEL_ID] = true;
    }
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
    LOG_DEBUG(HYBRID_BLOCKING + "start notify channelId {} pythonBatchId {} hybridBatchId {}",
        channelId, pythonBatchId[lastRunChannelId], hybridBatchId[channelId]);

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
    LOG_INFO(HYBRID_BLOCKING + "check step invalid, wait {} {}", channelId, hybridBatchId[channelId]);
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

void HybridMgmtBlock::CountPythonStep(int channelId, int steps)
{
    // 相应的通知计数
    pythonBatchId[channelId] += steps;
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
        LOG_DEBUG(HYBRID_BLOCKING + "The data channel was called for the first time, and the parameters were "
            "checked to be normal channelId {} hybridBatchId {}", channelId, hybridBatchId[channelId]);

        lastRunChannelId = channelId;
        return;
    }
    // 在通道切换时，hybrid预处理的batch与python的一致。
    if (pythonBatchId[lastRunChannelId] == hybridBatchId[lastRunChannelId]) {
        LOG_DEBUG(HYBRID_BLOCKING +
            "HybridMgmt is switching data channels and checking for normal parameters. he number of steps "
            "in the previous round is lastRunChannelId {} pythonBatchId {} hybridBatchId {}",
            lastRunChannelId,  pythonBatchId[lastRunChannelId], hybridBatchId[lastRunChannelId]);
    } else if (pythonBatchId[lastRunChannelId] < hybridBatchId[lastRunChannelId]) {
        // 在通道切换时，上一个通道处理的数据超出了python侧的调用
        if (!rankInfo.noDDR and !WaitValid(lastRunChannelId)) {
            throw HybridMgmtBlockingException("when channel switch");
        }
    } else {
        // 在通道切换时，hybrid处理的数据还没有赶上python侧，此时需要等待hybrid处理完成
        LOG_INFO(HYBRID_BLOCKING +
            "When switching data channels, it was found that HybridMgmt processed less data than the "
            "Python side.In this case, after reading the dataset, the Python side called it again, but it was "
            "interrupted midway,which did not affect the subsequent calls lastRunChannelId {} hybridBatchId {}",
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
    LOG_DEBUG(HYBRID_BLOCKING + "HybridMgmt starts blocking channelId {} hybridBatchId {}",
        channelId, hybridBatchId[channelId]);

    while (isBlock[channelId]) {
        std::this_thread::sleep_for(SLEEP_MS);
        if (!isRunning) {
            return;
        }
    }
    LOG_DEBUG(HYBRID_BLOCKING + "HybridMgmt is starting to wake up channelId {} hybridBatchId {}",
        channelId, hybridBatchId[channelId]);
    return;
}

/// 重置所有的步数，主要用于图重构的情况，readembedkey算子重建
/// \param channelId channelId  train 0 eval 1
void HybridMgmtBlock::ResetAll(int channelId)
{
    LOG_DEBUG(HYBRID_BLOCKING + "Hybridmgmt is resetting data channelId {} hybridBatchId {}",
        channelId, hybridBatchId[channelId]);

    readEmbedBatchId[channelId] = 0;
    pythonBatchId[channelId] = 0;
    hybridBatchId[channelId] = 0;
    isBlock[channelId] = false;
}

/// 检查当前的步数是否可以进行save
/// \return 0 is legal, 1 需要回退一步, -1 表示错误
int HybridMgmtBlock::CheckSaveEmbdMapValid()
{
    // 检查数据通道此时的HashMap是否被提前处理了
    if (pythonBatchId[lastRunChannelId] >= hybridBatchId[lastRunChannelId]) {
        LOG_DEBUG(HYBRID_BLOCKING +
            "HybridMgmt is checking the step and checking that the parameters are normal. "
            "The number of steps in the previous round is "
            "lastRunChannelId {} pythonBatchId {} hybridBatchId {}",
            lastRunChannelId, pythonBatchId[lastRunChannelId], hybridBatchId[lastRunChannelId]);
        return 0;
    } else if (pythonBatchId[lastRunChannelId] + 1 == hybridBatchId[lastRunChannelId]) {
        // 在通道切换时，上一个通道处理的数据超出了python侧的调用
        LOG_DEBUG(HYBRID_BLOCKING +
            "HybridMgmt is checking the step, and the parameters have been processed one step "
            "in advance. The number of steps in the previous round was "
            "lastRunChannelId {} pythonBatchId {} hybridBatchId {}",
            lastRunChannelId, pythonBatchId[lastRunChannelId], hybridBatchId[lastRunChannelId]);

        return 1;
    } else {
        // 在通道切换时，hybrid处理的数据还没有赶上python侧，此时需要等待hybrid处理完成
        LOG_DEBUG(HYBRID_BLOCKING + "ERROR FLAG lastRunChannelId {} hybridBatchId {}",
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

void HybridMgmtBlock::Destroy()
{
    if (!isRunning) {
        // 已经销毁过了，不用再次销毁会报错
        return;
    }
    isRunning = false;
}


void HybridMgmtBlock::SetRankInfo(RankInfo rankInfo)
{
    this->stepsInterval[TRAIN_CHANNEL_ID] = rankInfo.maxStep[TRAIN_CHANNEL_ID];
    this->stepsInterval[EVAL_CHANNEL_ID] = rankInfo.maxStep[EVAL_CHANNEL_ID];
    this->saveInterval = rankInfo.maxStep[SAVE_STEP_INDEX];
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