/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: hybrid mgmt module,Record the number of program running steps,
 * manage blocking and wakeup
 * Author: MindX SDK
 * Date: 2023/08/15
 */
#ifndef MX_REC_HYBRID_BLOCKING_H
#define MX_REC_HYBRID_BLOCKING_H
#include <chrono>
#include <thread>

#include "hd_transfer/hd_transfer.h"
#include "utils/common.h"
#include "utils/singleton.h"
using namespace MxRec;
const std::string HYBRID_BLOCKING = "[HYBRID_BLOCKING] ";
const std::string D2H_CHANNEL_NAME_PRE = "d2h_notify_hybridmgmt_";
const std::chrono::milliseconds SLEEP_MS = 20ms;
class HybridMgmtBlock {
public:
    // 上一次运行的通道ID
    int lastRunChannelId = -1;
    // hybrid将要处理的batch id
    int hybridBatchId[2] = {0, 0};
    // python侧将要处理的batch id
    int pythonBatchId[2] = {0, 0};
    // readEmbed算子侧将要处理的batch id
    int readEmbedBatchId[2] = {0, 0};
    bool isRunning = true;
    // 每个sparse lookup都会生成一个唯一的id，保证每次运行只有一个id在进行计数
    int uniqueSparseLookID[2]{-1, -1};

    ~HybridMgmtBlock();
    void CheckAndNotifyWake(int channelId);
    void CountPythonStep(int channelId);
    void CheckAndSetBlock(int channelId);
    void CheckValid(int channelId);
    void DoBlock(int channelId);
    void ResetAll(int channelId);
    int CheckSaveEmbdMapValid();
    bool GetBlockStatus(int channelId);
    void SetBlockStatus(int channelId, bool block);
    void SetRankInfo(RankInfo rankInfo);
    void SetStepInterval(int trainStep, int evalStep);
    void StartNotifySignalMonitor();
    bool WaitValid(int channelId);
    void Destroy();
private:
    // 通道i运行多少步后切换为通道j
    int stepsInterval[2] = {0, 0};
     // 控制通道阻塞的变量
    bool isBlock[2] = {true, true};
    string d2hChannelName[2];
    RankInfo rankInfo;
    acltdtChannelHandle* aclHandles[2];
    std::vector<std::unique_ptr<std::thread>> procThreads {};
};

class HybridMgmtBlockingException : public std::exception {
public:
    explicit HybridMgmtBlockingException(const string scene)
    {
        HybridMgmtBlock* hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
        // int channelId, int preprocessBatchNumber, int currentBatchNumber
        int channelId = hybridMgmtBlock->lastRunChannelId;
        int preprocessBatchNumber = hybridMgmtBlock->hybridBatchId[channelId];
        int currentBatchNumber = hybridMgmtBlock->pythonBatchId[channelId];
        str = StringFormat("Error happened at HyBridmgmt Blocking, it finds that "
                           "preprocess batch number not match current using batch number "
                           "%s , last use channel id is %d, preprocessBatchNumber is %d ,"
                           "currentBatchNumber is %d. please check your setting of train "
                           "steps and eval steps", scene.c_str(), channelId, preprocessBatchNumber,
                           currentBatchNumber);
        LOG(ERROR) << str;
    }

private:
    string str;
};
#endif