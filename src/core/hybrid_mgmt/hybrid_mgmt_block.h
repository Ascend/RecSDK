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

#ifndef MX_REC_HYBRID_BLOCKING_H
#define MX_REC_HYBRID_BLOCKING_H

#include <chrono>
#include <thread>

#include "hd_transfer/hd_transfer.h"
#include "utils/common.h"
#include "utils/singleton.h"

namespace MxRec {
    const std::string HYBRID_BLOCKING = "[HYBRID_BLOCKING] ";
    const int SAVE_STEP_INDEX = 2;
    const std::chrono::milliseconds SLEEP_MS = 20ms;

    class HybridMgmtBlock {
    public:
        HybridMgmtBlock() = default;
        // 上一次运行的通道ID
        int lastRunChannelId = -1;
        // hybrid将要处理的batch id
        int hybridBatchId[2] = {0, 0};
        // python侧将要处理的batch id
        int pythonBatchId[2] = {0, 0};
        // readEmbed算子侧将要处理的batch id
        int readEmbedBatchId[2] = {0, 0};

        int loop[2] = {1, 1};

        bool isRunning = true;

        HybridMgmtBlock(const HybridMgmtBlock&) = delete;

        HybridMgmtBlock& operator=(const HybridMgmtBlock&) = delete;

        ~HybridMgmtBlock();

        void CheckAndNotifyWake(int channelId);

        void CountPythonStep(int channelId, int steps);

        void CheckAndSetBlock(int channelId);

        void CheckValid(int channelId);

        void DoBlock(int channelId);

        void ResetAll(int channelId);

        int CheckSaveEmbMapValid();

        bool GetBlockStatus(int channelId);

        void SetBlockStatus(int channelId, bool block);

        void SetRankInfo(RankInfo ri);

        void SetStepInterval(int trainStep, int evalStep);

        bool WaitValid(int channelId);

        void Destroy();

    private:
        // 通道i运行多少步后切换为通道j
        int stepsInterval[2] = {0, 0};
        // 控制通道阻塞的变量
        bool isBlock[2] = {true, true};
        // 控制训练了多少步进行保存的步数
        int saveInterval = 0;
        RankInfo rankInfo;
    };

    class HybridMgmtBlockingException : public std::exception {
    public:
        explicit HybridMgmtBlockingException(const string scene)
        {
            HybridMgmtBlock *hybridMgmtBlock = Singleton<HybridMgmtBlock>::GetInstance();
            int channelId = hybridMgmtBlock->lastRunChannelId;
            int preprocessBatchNumber = hybridMgmtBlock->hybridBatchId[channelId];
            int currentBatchNumber = hybridMgmtBlock->pythonBatchId[channelId];
            str = StringFormat("Error happened at HyBridmgmt Blocking, it finds that "
                               "preprocess batch number not match current using batch number "
                               "%s , last use channel id is %d, preprocessBatchNumber is %d ,"
                               "currentBatchNumber is %d. please check your setting of train "
                               "steps and eval steps", scene.c_str(), channelId, preprocessBatchNumber,
                               currentBatchNumber);
            LOG_ERROR(str);
        }

    private:
        string str;
    };
}
#endif