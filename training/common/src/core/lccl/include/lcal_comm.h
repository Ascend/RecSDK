/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef LCAL_COMM_H
#define LCAL_COMM_H

#ifdef GTEST
#define GTEST_PRIVATE public
#define TEST_VIRTUAL virtual
#else
#define GTEST_PRIVATE private
#define TEST_VIRTUAL
#endif

#include <vector>
#include <string>
#include <unordered_map>

#include <hccl.h>

#include "lcal_types.h"
#include "comm_args.h"
#include "lcal_api.h"

namespace Lcal {
constexpr int IPC_NAME_SIZE = 65;

class LcalSockExchange;
class LcalComm {
public:
    LcalComm(int rank, int rankSize);
    LcalComm(int rank, int rankSize, std::vector<int>& rankList);
    LcalComm(int rank, int rankSize, LcalUniqueId commId);
    ~LcalComm();
    LcalComm(const LcalComm&) = delete;
    LcalComm &operator=(const LcalComm&) = delete;
    int Init();
    int InitThread();
    int GetRank() const;
    int GetRankSize() const;
    int GetCommSize() const;
    const PhysicalInfo& GetPhysicalInfo() const;
    int8_t* GetCommArgsPtr();
    friend class Lccl;
    friend class Lcoc;

    int rank_ = 0;  // global rank id
    int rankSize_ = 0;  // global rank size
    int commSize_ = 0;  // local LcalComm size
    int localRank_ = -1;
    int localRankSize_ = -1;
    int devId_ = 0;
    int64_t magic_ = 1;
    bool inited_ = false;
    std::vector<int> devList_ = {};
    std::vector<int> rankList_ = {};
    // Shared ping pong buff，allocate on HBM, host can access but not allow to modify directly.
    int8_t* peerMem_[LCAL_MAX_RANK_SIZE] = {};
    PhysicalInfo physicalInfo_ = {};
    CommArgs commArgs_ = {};  // host
    int8_t* commArgsPtr_ = nullptr;  // device
    LcalUniqueId commId_ = {};
    LcalSockExchange* socketExchange_ = nullptr;

GTEST_PRIVATE:
    virtual int SetMemoryName(std::string& name);
    virtual int SetIpcPidSdid(std::string& name, const uint32_t* pids, const int64_t* sdids) const;
    virtual int OpenIpcMem(const char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE]);
    virtual int GatherDevId();
    virtual int GatherDevIdThread();
    virtual int EnablePeerAccess();
    virtual int InitCommMem();
    virtual int InitCommon();
    TEST_VIRTUAL void CloseIpcMem();
    TEST_VIRTUAL void FreePeerMem(int8_t*& mem);
    virtual int InitMem();
    virtual int GetSidId(int64_t sdids[LCAL_MAX_RANK_SIZE]);
    virtual int GetPid(uint32_t pids[LCAL_MAX_RANK_SIZE]);
    virtual int GetName(std::string& name, char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE]);
    virtual int SyncCommArgs();
};

uint32_t GetCoreNum(ChipName chipName);

ChipName& GetChipName();

bool SkipUnusedChannel910B2C(int curRank, int peerRank, ChipName chipName);

} // namespace Lcal

#endif // LCAL_COMM_H
