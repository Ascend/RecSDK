/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "host_mgmt.h"
#include "hd_transfer/rma_shm_svm.h"
#include "utils/config.h"
#include "utils/logger.h"
#include "utils/error.h"
#include "rma_swap/rma_swap_test.h"
#include "tdt_transfer/tdt_transfer.h"

using namespace std;

namespace TfTest {

bool HostMgmt::Initialize(vector <ShmInfo>& shmInfos, int stepNum, int seed, string type)
{
    MxRec::ConfigGlobalEnv();

    if (isRunning) {
        return true;
    }

    mgmtShmInfos = shmInfos;
    this->stepNum = stepNum;
    this->seed = seed;
    this->type = type;

    auto success = CreateChannel(shmInfos);
    if (!success) {
        return false;
    }
    if (type == "RMA") {
        RmaSwapTask();
    } else {
        size_t dataSize = shmInfos[0].shape[0] * shmInfos[0].shape[1] * sizeof(float);
        if (aclrtMallocHost((void **)&tdtBuffer, dataSize) != ACL_ERROR_NONE) {
            LOG_ERROR("Malloc tdt send data failed.");
            throw runtime_error("Malloc tdt send data failed.");
        }
        (void)aclrtMemset(tdtBuffer, dataSize, 1, dataSize);
        TdtSwapTask();
    }

    isRunning = true;

    return true;
}

bool HostMgmt::CreateChannel(vector <ShmInfo>& shmInfos)
{
    for (auto& shmInfo: shmInfos) {
        for (int c = static_cast<int>(TransferChannel::D2H); c <= static_cast<int>(TransferChannel::H2D); c++) {
            auto channel = static_cast<TransferChannel>(c);
            string channelName = shmInfo.channelName + "_" + TransferChannel2Str(channel);
            if (type == "RMA") {
                auto addr = MxRec::GetShmAddr(channelName, shmInfo.deviceId, shmInfo.capacity);
                if (addr == 0) {
                    LOG_ERROR("Malloc SHM failed!");
                    return false;
                }
            } else if (type == "TDT") {
                CreateTdtChannel(channelName, shmInfo.deviceId, shmInfo.capacity); // 返回值校验
            } else {
                LOG_ERROR("Unsupported transfer type {}.", type.c_str());
                return false;
            }
        }
    }
    return true;
}

void HostMgmt::Destroy()
{
    JoinRmaSwapThread();
    if (mgmtShmInfos.size() != 0) {
        if (type == "RMA") {
            MxRec::FreeShmAddr(mgmtShmInfos[0].deviceId);
        } else if (type == "TDT") {
            FreeTdtChannel(mgmtShmInfos[0].deviceId);
            if (aclrtFreeHost(tdtBuffer) != ACL_ERROR_NONE) {
                LOG_WARN("Release tdtBuffer failed.");
            }
        }
    }
    mgmtShmInfos.clear();
    isRunning = false;
}

void HostMgmt::RmaSwapTask()
{
    for (auto& shmInfo: mgmtShmInfos) {
        auto swapInTask = [&]() {
            string channelName = shmInfo.channelName + "_" + TransferChannel2Str(TransferChannel::H2D);
            for (int step = 0; step < stepNum; ++step) {
                if (WriteShmData(channelName, shmInfo.shape) == 0) {
                    LOG_WARN("H2D[{}] failed.", step);
                    return;
                }
                LOG_INFO("H2D[{}] success.", step);
            }
        };
        swapInThreads.emplace_back(swapInTask);

        auto swapOutTask = [&]() {
            string channelName = shmInfo.channelName + "_" + TransferChannel2Str(TransferChannel::D2H);
            for (int step = 0; step < stepNum; ++step) {
                if (ReadShmData(channelName) == 0) {
                    LOG_WARN("D2H[{}] failed.", step);
                    return;
                }
                LOG_INFO("D2H[{}] success.", step);
            }
        };
        swapOutThreads.emplace_back(swapOutTask);
    }
}

void HostMgmt::TdtSwapTask()
{
    for (auto& shmInfo: mgmtShmInfos) {
        auto swapInTask = [&]() {
            string channelName = shmInfo.channelName + "_" + TransferChannel2Str(TransferChannel::H2D);
            for (int step = 0; step < stepNum; ++step) {
                size_t dataSize = shmInfo.shape[0] * shmInfo.shape[1] * sizeof(float);
                int64_t dims[2] = {shmInfo.shape[0], shmInfo.shape[1]};
                SendByAclTdtV2(channelName, (float *)tdtBuffer, dataSize, dims);
                LOG_INFO("H2D[{}] success.", step);
            }
        };
        swapInThreads.emplace_back(swapInTask);

        auto swapOutTask = [&]() {
            string channelName = shmInfo.channelName + "_" + TransferChannel2Str(TransferChannel::D2H);
            for (int step = 0; step < stepNum; ++step) {
                RecvByChannel(channelName, 1);
                LOG_INFO("D2H[{}] success.", step);
            }
        };
        swapOutThreads.emplace_back(swapOutTask);
    }
}

void HostMgmt::JoinRmaSwapThread()
{
    for (auto& t: swapInThreads) {
        t.join();
    }
    swapInThreads.clear();
    for (auto& t: swapOutThreads) {
        t.join();
    }
    swapOutThreads.clear();
}
}