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
#include <lcal_comm.h>

#include <chrono>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <set>
#include <thread>

#include <hccl.h>

#include "asdops/utils/log/log.h"
#include "tools/socket/lcal_sock_exchange.h"

#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/dev.h"

using namespace std;
using namespace chrono;
using namespace AsdOps;

namespace Lcal {

constexpr int AI_CORE_NUM_24 = 24;
constexpr int AI_CORE_NUM_20 = 20;
constexpr int AI_CORE_NUM_2 = 2;

enum TopologyType : int {
    TOPOLOGY_HCCS = 0,
    TOPOLOGY_PIX,
    TOPOLOGY_PIB,
    TOPOLOGY_PHB,
    TOPOLOGY_SYS,
    TOPOLOGY_SIO,
    TOPOLOGY_HCCS_SW
};

constexpr int HCCL_IPC_PID_ARRAY_SIZE = 1; // 固定每次只传一个PID数据
constexpr int LCAL_INIT_TIMEOUT = 600;


static const std::unordered_map<std::string, ChipName> chipMap = {
    {"Ascend310P", ChipName::CHIP_310P3},
    {"Ascend910B1", ChipName::CHIP_910B1},
    {"Ascend910B2", ChipName::CHIP_910B2},
    {"Ascend910B2C", ChipName::CHIP_910B2C},
    {"Ascend910B3", ChipName::CHIP_910B3},
    {"Ascend910B4", ChipName::CHIP_910B4},
    {"Ascend910B4-1", ChipName::CHIP_910B41},
    {"Ascend910_9391", ChipName::CHIP_910_9391},
    {"Ascend910_9381", ChipName::CHIP_910_9381},
    {"Ascend910_9392", ChipName::CHIP_910_9392},
    {"Ascend910_9382", ChipName::CHIP_910_9382},
    {"Ascend910_9372", ChipName::CHIP_910_9372},
    {"Ascend910_9362", ChipName::CHIP_910_9362},
    {"Ascend910_9361", ChipName::CHIP_910_9361},
};

/**
 * @brief 用于获取芯片名称
 */
ChipName& GetChipName()
{
    // 在分配内存时用到
    static ChipName curChipName = ChipName::RESERVED;
    if (curChipName != ChipName::RESERVED) {
        return curChipName;
    }
    constexpr int socVerLength = 100; // asd没有相应的宏和常量，这里和asd测试代码中的长度保持一致
    char ver[socVerLength];
    auto ret = rtGetSocVersion(ver, socVerLength);
    if (ret != 0) {
        ASD_LOG(WARN) << "rtGetSocVersion failed.";
        return curChipName;
    }
    string chipName(ver);
    ASD_LOG(DEBUG) << "rtGetSocVersion -- :" << chipName;

    auto it = chipMap.find(chipName);
    if (it != chipMap.end()) {
        curChipName = it->second;
    } else {
        ASD_LOG(WARN) << "Unkown chip name.";
    }
    return curChipName;
}

uint32_t GetCoreNum(ChipName chipName)
{
    switch (chipName) {
        case ChipName::CHIP_910B1:
        case ChipName::CHIP_910B2:
        case ChipName::CHIP_910_9391:
        case ChipName::CHIP_910_9381:
        case ChipName::CHIP_910_9392:
        case ChipName::CHIP_910_9382:
        case ChipName::CHIP_910B2C:
            return AI_CORE_NUM_24;
        case ChipName::CHIP_910B3:
        case ChipName::CHIP_910B4:
        case ChipName::CHIP_910B41:
        case ChipName::CHIP_910_9372:
        case ChipName::CHIP_910_9362:
        case ChipName::CHIP_910_9361:
            return AI_CORE_NUM_20;
        case ChipName::CHIP_310P3:
            return AI_CORE_NUM_2;
        default:
            ASD_LOG(ERROR) << "Unknown chip name";
            return 0;
    }
}

// 如果是互联的链路，返回false； 对910B2C那些不互联的链路，返回true
bool SkipUnusedChannel910B2C(int curRank, int peerRank, ChipName chipName)
{
    if (chipName == ChipName::CHIP_910B2C) {
        constexpr int rankSizePerNode = 8;
        // 双节点16P中不用的链路: 不在同一个节点 且rank在节点内序号不同； 在调用时将跳过
        if ((curRank / rankSizePerNode != peerRank / rankSizePerNode) &&
            (std::abs(curRank - peerRank) != rankSizePerNode)) {
            return true;
        }
    }
    return false;
}

int LcalComm::SyncCommArgs()
{
    commArgs_.rank = rank_;
    commArgs_.localRank = (localRank_ == -1) ? rank_ : localRank_;
    commArgs_.rankSize = rankSize_;
    commArgs_.localRankSize = (localRankSize_ == -1) ? rankSize_ : localRankSize_;
    commArgs_.SetBuff(peerMem_);
    int ret = aclrtMalloc(reinterpret_cast<void **>(&commArgsPtr_), sizeof(commArgs_), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ASD_LOG(ERROR) << "aclrtMalloc err " << __LINE__ << " " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    ret = aclrtMemcpy(commArgsPtr_, sizeof(commArgs_), &commArgs_, sizeof(commArgs_), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        ASD_LOG(ERROR) << "aclrtMemcpy err " << __LINE__ << " " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

int LcalComm::InitCommon()
{
    // enable peer device
    if (EnablePeerAccess() != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "EnablePeerAccess failed!";
        return LCAL_ERROR_INTERNAL;
    }

    localRank_ = rank_ % localRankSize_;
    return LCAL_SUCCESS;
}

void LcalComm::CloseIpcMem()
{
    for (int i = 0; i < rankSize_; ++i) {
        if (i == rank_ || peerMem_[i] == nullptr) {
            continue;
        }
        int ret = rtIpcCloseMemory(reinterpret_cast<void*>(peerMem_[i]));
        if (ret != 0) {
            ASD_LOG(WARN) << "Close ipc[" << i << "] memory failed! ret: " << ret;
        }
        peerMem_[i] = nullptr;
    }
}

void LcalComm::FreePeerMem(int8_t*& mem)
{
    if (mem != nullptr) {
        aclError aclRet = aclrtFree(reinterpret_cast<void*>(mem));
        if (aclRet != ACL_SUCCESS) {
            ASD_LOG(ERROR) << "Free share memory failed! ret: " << aclRet;
        }
    }
    mem = nullptr;
}

int LcalComm::Init()
{
    if (inited_) {
        return LCAL_SUCCESS;
    }
    if (rank_ < 0 || rank_ >= rankSize_ || rankSize_ <= 0 || rankSize_ > LCAL_MAX_RANK_SIZE) {
        ASD_LOG(ERROR) << "The rank is invalid! rank:" << rank_ << " rankSize:" << rankSize_;
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }
    if (LcalSockExchange::CheckValid(commId_)) {
        socketExchange_ = new LcalSockExchange(rank_, rankSize_, commId_);
    } else {
        socketExchange_ = new LcalSockExchange(rank_, rankSize_, rankList_);
    }
    int ret = GatherDevId();
    if (ret != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "init context failed! ret: " << ret;
        return ret;
    }

    ASD_LOG(INFO) << "rank " << rank_ << "/" << rankSize_ << " running devId:" << devId_;

    if (InitCommon() != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "init common failed!";
        return LCAL_ERROR_INTERNAL;
    }

    ASD_LOG(DEBUG) << "Start InitCommMem localRankSize_ -> " << localRankSize_ << ", localRank_ -> " << localRank_;
    if (InitCommMem() != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "InitCommMem failed!";
        return LCAL_ERROR_INTERNAL;
    }
    ASD_LOG(DEBUG) << "InitCommMem " << rank_ << "/" << rankSize_ << ", localRank_ : " << localRank_ <<
            ", localRankSize_ : " << localRankSize_ << " success";

    // set comm args in device.
    SyncCommArgs();
    ASD_LOG(INFO) << "LcalCommInit " << rank_ << "/" << rankSize_ << " success and extraFlag:" << commArgs_.extraFlag <<
        " commArgs_.localRank : " << commArgs_.localRank << " commArgs_.localRankSize : " << commArgs_.localRankSize;
    inited_ = true;
    delete socketExchange_;
    socketExchange_ = nullptr;
    return LCAL_SUCCESS;
}

int LcalComm::InitThread()
{
    if (inited_) {
        return LCAL_SUCCESS;
    }
    if (rank_ < 0 || rank_ >= rankSize_ || rankSize_ <= 0 || rankSize_ > LCAL_MAX_RANK_SIZE) {
        ASD_LOG(ERROR) << "The rank is invalid! rank:" << rank_ << "rankSize:" << rankSize_;
        return LCAL_ERROR_PARA_CHECK_FAIL;
    }

    if (GatherDevIdThread() != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "get devs failed.";
        return LCAL_ERROR_INTERNAL;
    }
    ASD_LOG(INFO) << "rank " << rank_ << "/" << rankSize_ << " running devId:" << devId_;

    if (InitCommon() != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "init common failed!";
        return LCAL_ERROR_INTERNAL;
    }
    static int8_t *localPeerMem[LCAL_MAX_RANK_SIZE] = {};
    aclError aclRet = aclrtMalloc(
        reinterpret_cast<void **>(&localPeerMem[rank_]), LCAL_BUFF_BYTES,
        (GetChipName() == ChipName::CHIP_310P3) ? ACL_MEM_MALLOC_HUGE_FIRST_P2P : ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        ASD_LOG(ERROR) << "InitCommMem failed! ret: " << aclRet;
        return LCAL_ERROR_INTERNAL;
    }
    auto start = high_resolution_clock::now();
    for (int i = 0; i < rankSize_; ++i) {
        while (localPeerMem[i] == nullptr) {  // check other threads
            this_thread::sleep_for(1ms);
            auto elapsed = duration_cast<seconds>(high_resolution_clock::now() - start);
            if (elapsed.count() > LCAL_INIT_TIMEOUT) {
                ASD_LOG(ERROR) << "LCCL Init timeout!";
                FreePeerMem(localPeerMem[rank_]);
                return LCAL_ERROR_TIMEOUT;
            }
        }
        peerMem_[i] = localPeerMem[i];
    }
    SyncCommArgs();
    ASD_LOG(INFO) << "LCCL init multi thread " << rank_ << "/" << rankSize_ << " success";
    inited_ = true;
    return LCAL_SUCCESS;
}

/**
 * @brief 函数内部会有检测，是否需要进行 aclrtDeviceEnablePeerAccess，如果芯片为310P且是HCCS链路，则不调用此函数。
 *
 *
 */
int LcalComm::EnablePeerAccess()
{
    physicalInfo_.chipName = GetChipName();
    for (auto &dev : devList_) {
        if (devId_ == dev) {
            continue;
        }
        // 处理910B2C 16卡通信的特例
        if (SkipUnusedChannel910B2C(dev, devId_, GetChipName())) {
            continue;
        }

        int64_t value = 0;
        if (rtGetPairDevicesInfo(devId_, dev, 0, &value) != 0) {
            ASD_LOG(WARN) << devId_ << " and " << dev << " get pair info fail.";
        } else {
            ASD_LOG(DEBUG) << devId_ << " <-----> " << dev << ", halGetPairDevicesInfo: *value = " << value;
        }

        // 如果310P未来通信域要支持两卡四芯的话，这里需要做更改。并且现在默认服务器上机器只有一个链路种类。
        if (value == TOPOLOGY_HCCS || value == TOPOLOGY_SIO || value == TOPOLOGY_HCCS_SW) {
            physicalInfo_.physicalLink = PhysicalLink::HCCS;
        } else if (physicalInfo_.physicalLink == PhysicalLink::RESERVED) {
            physicalInfo_.physicalLink = PhysicalLink::PCIE;
            if (rankSize_ > PING_PONG_SIZE) {
                ASD_LOG(ERROR) << "do not support pcie > 2 rank! rankSize_ = " << rankSize_;
                return LCAL_ERROR_INTERNAL;
            }
        }

        physicalInfo_.coreNum = GetCoreNum(physicalInfo_.chipName);

        // value里的0实际上对应驱动枚举类的 TOPOLOGY_HCCS
        if (physicalInfo_.chipName == ChipName::CHIP_310P3 && value == 0) {
            ASD_LOG(WARN) << "aclrtDeviceEnablePeerAccess is skipped! peerDeviceId = " << dev;
            continue;
        }

        aclError ret = aclrtDeviceEnablePeerAccess(dev, 0);
        if (ret != ACL_SUCCESS) {
            ASD_LOG(ERROR) << "aclrtDeviceEnablePeerAccess failed peerDeviceId = " << dev << " ,rank = " << rank_
                           << ", value = " << value << ", flags = " << 0 << "," << __LINE__ << ": " << ret;
            return LCAL_ERROR_INTERNAL;
        }
    }
    ASD_LOG(DEBUG) << "EnablePeerAccess succeed" << rank_;
    return LCAL_SUCCESS;
}

int LcalComm::GatherDevId()
{
    int nodeNum = socketExchange_->GetNodeNum();
    if (nodeNum <= 0 or nodeNum > rankSize_) {
        ASD_LOG(ERROR) << "error! node num : " << nodeNum << " rank size: " << rankSize_;
        return LCAL_ERROR_INTERNAL;
    }
    localRankSize_ = rankSize_ / nodeNum;
    localRank_ = rankSize_ % nodeNum;
    devList_.resize(rankSize_);
    // get current id and broadcast
    aclError aclRet = aclrtGetDevice(&devId_);
    if (aclRet != ACL_SUCCESS) {
        ASD_LOG(ERROR) << "aclrtGetDevice error! ret: " << aclRet;
        return LCAL_ERROR_INTERNAL;
    }
    // get other rank dev id, put into devList_
    int ret = socketExchange_->AllGather(&devId_, sizeof(devId_), devList_.data());
    if (ret != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "LcalSockExchange AllGather error! ret: " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    std::string devIdStr = "";
    for (int i = 0; i < rankSize_; ++i) {
        devIdStr += (i == 0 ? "" : ", ");
        devIdStr += to_string(devList_[i]);
    }
    ASD_LOG(DEBUG) << "rank " << rank_ << " devId: " << devId_ << ", otherDevList : " << devIdStr;
    ASD_LOG(INFO) << "AllGather: Get other rank dev id success";
    return LCAL_SUCCESS;
}

int LcalComm::GatherDevIdThread()
{
    devList_.resize(rankSize_);
    // get current id and broadcast
    aclError aclRet = aclrtGetDevice(&devId_);
    if (aclRet != ACL_SUCCESS) {
        aclRet = aclrtSetDevice(rank_);
        if (aclRet != ACL_SUCCESS) {
            ASD_LOG(ERROR) << "aclrtSetDevice error! ret: " << aclRet;
            return LCAL_ERROR_INTERNAL;
        }
        devId_ = rank_;
        ASD_LOG(WARN) << "aclrtGetDevice error, but aclrtSetDevice success.";
    }
    static int devList[LCAL_MAX_RANK_SIZE];
    devList[rank_] = devId_ + 1;  // 0 is invalid
    auto start = high_resolution_clock::now();
    for (int i = 0; i < rankSize_; ++i) {
        while (devList[i] == 0) {  // check other threads
            this_thread::sleep_for(1ms);
            auto elapsed = duration_cast<seconds>(high_resolution_clock::now() - start);
            if (elapsed.count() > LCAL_INIT_TIMEOUT) {
                ASD_LOG(ERROR) << "LCCL Init timeout!";
                return LCAL_ERROR_TIMEOUT;
            }
        }
        devList_.at(i) = devList[i] - 1;
    }
    return LCAL_SUCCESS;
}

int LcalComm::InitMem()
{
    // 申请并初始化IpcBuff
    ASD_LOG(DEBUG) << "maxBuffSize " << LCAL_BUFF_BYTES;
    int memRank = (GetChipName() < ChipName::CHIP_910_9391) ? localRank_ : rank_;
    aclError ret = aclrtMalloc(
        reinterpret_cast<void **>(&peerMem_[memRank]), LCAL_BUFF_BYTES,
        (GetChipName() == ChipName::CHIP_310P3) ? ACL_MEM_MALLOC_HUGE_FIRST_P2P : ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        ASD_LOG(ERROR) << "allocate device mem error " << __FILE__ << ":" << __LINE__ << " " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    ASD_LOG(DEBUG) << "peerMem[rank" << rank_ << "], allocate finished.";
    aclrtMemset(peerMem_[memRank], LCAL_BUFF_BYTES, 0, LCAL_BUFF_BYTES);
    return LCAL_SUCCESS;
}

int LcalComm::GetPid(uint32_t pids[LCAL_MAX_RANK_SIZE])
{
    if (rtDeviceGetBareTgid(&pids[rank_]) != 0) {  // 获取docker外的进程id，bare指docker外
        ASD_LOG(ERROR) << "DeviceGetBareTgid err " << __LINE__;
        return LCAL_ERROR_INTERNAL;
    }
    int ret = socketExchange_->AllGather(&pids[rank_], sizeof(pids[rank_]), pids);
    if (ret != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "LcalSockExchange AllGather error! ret: " << ret;
        return ret;
    }
    for (int i = 0; i < rankSize_; ++i) {
        ASD_LOG(DEBUG) << "rank " << i << " pid: " << pids[i];
    }
    ASD_LOG(DEBUG) << "AllGather: Get other rank pid";
    return LCAL_SUCCESS;
}

int LcalComm::GetSidId(int64_t sdids[LCAL_MAX_RANK_SIZE])
{
    if ((physicalInfo_.chipName >= ChipName::CHIP_910_9391) && (physicalInfo_.chipName < ChipName::RESERVED)) {
        const int rtModuleTypeSystem = 0;
        const int infoTypeSdid = 26;
        if (rtGetDeviceInfo(devList_[rank_], rtModuleTypeSystem, infoTypeSdid, &sdids[rank_]) != 0) {
            ASD_LOG(ERROR) << "DeviceGetDeviceInfo err " << __LINE__;
            return LCAL_ERROR_INTERNAL;
        }
        ASD_LOG(DEBUG) << "rank " << rank_ << " dev id: " << devList_[rank_]
                       << " rtGetDeviceInfo sdid: " << sdids[rank_];

        int ret = socketExchange_->AllGather(&sdids[rank_], sizeof(sdids[rank_]), sdids);
        if (ret != LCAL_SUCCESS) {
            ASD_LOG(ERROR) << "LcalSockExchange AllGather error! ret: " << ret;
            return ret;
        }
        for (int i = 0; i < rankSize_; ++i) {
            ASD_LOG(DEBUG) << "rank " << i << " sdid: " << sdids[i];
        }
        ASD_LOG(DEBUG) << "AllGather: Get other rank sdid";
    }
    return LCAL_SUCCESS;
}

int LcalComm::GetName(string &name, char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE])
{
    int ret = socketExchange_->AllGather(name.c_str(), IPC_NAME_SIZE, names);
    if (ret != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "LcalSockExchange AllGather error! ret: " << ret;
        return LCAL_ERROR_INTERNAL;
    }
    for (int i = 0; i < rankSize_; ++i) {
        ASD_LOG(DEBUG) << "rank " << i << " mem name: " << names[i];
    }
    ASD_LOG(DEBUG) << "AllGather: Get other rank mem name";
    return LCAL_SUCCESS;
}

int LcalComm::InitCommMem()
{
    int ret = InitMem();
    if (ret != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "InitMem error! ret: " << ret;
        return ret;
    }

    // 获取所有进程的pid
    uint32_t pids[LCAL_MAX_RANK_SIZE];
    ret = GetPid(pids);
    if (ret != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "GetPid error! ret: " << ret;
        return ret;
    }

    // 获取所有进程的sdid
    int64_t sdids[LCAL_MAX_RANK_SIZE];
    ret = GetSidId(sdids);
    if (ret != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "GetSidId error! ret: " << ret;
        return ret;
    }

    // 获取所有进程的mem name
    string name;
    if (SetMemoryName(name) != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "SetMemoryName err ";
        return LCAL_ERROR_INTERNAL;
    }

    if (SetIpcPidSdid(name, pids, sdids) != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "SetIpcPidSdid failed!";
        return LCAL_ERROR_INTERNAL;
    }

    ASD_LOG(DEBUG) << "rank:" << rank_ << ", peermem name:" << name << ", name len:" << name.size();
    char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE];
    name.resize(IPC_NAME_SIZE);
    ret = GetName(name, names);
    if (ret != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "GetName error! ret: " << ret;
        return ret;
    }

    if (OpenIpcMem(names) != LCAL_SUCCESS) {
        ASD_LOG(ERROR) << "rank: " << rank_ << " OpenIpcMem failed!";
        return LCAL_ERROR_INTERNAL;
    }
    return LCAL_SUCCESS;
}

int LcalComm::OpenIpcMem(const char names[LCAL_MAX_RANK_SIZE][IPC_NAME_SIZE])
{
    static mutex mut;
    lock_guard<mutex> lock(mut);
    ChipName chipName = GetChipName();
    if (chipName <= ChipName::CHIP_910B2C) {
        for (int i = 0; i < rankSize_; ++i) {
            if (i == rank_) {
                continue;
            }
            // 处理910B2C 16卡通信的特例
            if (SkipUnusedChannel910B2C(rank_, i, GetChipName())) {
                continue;
            }
            int ret = rtIpcOpenMemory(reinterpret_cast<void **>(&peerMem_[i % localRankSize_]), names[i]);
            if (ret != 0) {
                ASD_LOG(ERROR) << "rank : " << rank_ << " localRank : " << localRank_ << " peerMem: " << i <<
                    " IpcOpenMemory err " << ret;
                return LCAL_ERROR_INTERNAL;
            }
        }
        return LCAL_SUCCESS;
    }
    if (chipName < ChipName::RESERVED) {
        for (int i = 0; i < rankSize_; ++i) {
            if (i == rank_) {
                continue;
            }
            int ret = rtIpcOpenMemory(reinterpret_cast<void **>(&peerMem_[i]), names[i]);
            if (ret != 0) {
                ASD_LOG(ERROR) << "rank : " << rank_ << " localRank : " << localRank_ << " peerMem: " << i <<
                    " IpcOpenMemory err " << ret;
                return LCAL_ERROR_INTERNAL;
            }
        }
        return LCAL_SUCCESS;
    }
    return LCAL_ERROR_INTERNAL;
}

int LcalComm::SetMemoryName(string& name)
{
    char nameModified[IPC_NAME_SIZE];
    int memRank = (GetChipName() < ChipName::CHIP_910_9391) ? localRank_ : rank_;
    if (rtIpcSetMemoryName(peerMem_[memRank], LCAL_BUFF_BYTES, nameModified, IPC_NAME_SIZE) != 0) {
        return LCAL_ERROR_INTERNAL;
    }
    name = nameModified;
    return LCAL_SUCCESS;
}

int LcalComm::SetIpcPidSdid(string &name, const uint32_t *pids, const int64_t *sdids) const
{
    for (int i = 0; i < rankSize_; ++i) {
        if (i == rank_) {
            continue;
        }
        
        if (regularChip.find(physicalInfo_.chipName) != regularChip.end()) {
            // 910B
            int32_t pidInt32 = pids[i];
            int rtRet = rtSetIpcMemPid(name.c_str(), &pidInt32, HCCL_IPC_PID_ARRAY_SIZE);
            if (rtRet != 0) {
                ASD_LOG(ERROR) << "err " << rtRet;
                return LCAL_ERROR_INTERNAL;
            }
        } else {
            // 910_93
            int32_t pidInt32 = pids[i];
            int rtRet = rtSetIpcMemorySuperPodPid(name.c_str(), sdids[i], &pidInt32, HCCL_IPC_PID_ARRAY_SIZE);
            if (rtRet != 0) {
                ASD_LOG(ERROR) << "err " << rtRet;
                return LCAL_ERROR_INTERNAL;
            }
        }
    }
    return LCAL_SUCCESS;
}

LcalComm::~LcalComm()
{
    CloseIpcMem();
    FreePeerMem(peerMem_[rank_]);
    if (socketExchange_) {
        delete socketExchange_;
        socketExchange_ = nullptr;
    }
    FreePeerMem(commArgsPtr_);
}

LcalComm::LcalComm(int rank, int rankSize) : rank_(rank), rankSize_(rankSize)
{
}

LcalComm::LcalComm(int rank, int rankSize, std::vector<int> &rankList)
    : rank_(rank), rankSize_(rankSize), rankList_(rankList)
{
}

LcalComm::LcalComm(int rank, int rankSize, LcalUniqueId commId)
    : rank_(rank), rankSize_(rankSize), commId_(commId)
{
}

int LcalComm::GetRank() const
{
    return rank_;
}

int LcalComm::GetRankSize() const
{
    return rankSize_;
}

int LcalComm::GetCommSize() const
{
    return commSize_;
}

const PhysicalInfo &LcalComm::GetPhysicalInfo() const
{
    return physicalInfo_;
}

int8_t* LcalComm::GetCommArgsPtr()
{
    return commArgsPtr_;
}

}  // Lcal
