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
#include "rma_shm_svm.h"
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <sys/stat.h>
#include <sys/shm.h>
#include <acl/acl.h>
#include <driver/ascend_hal_define.h>
#include "securec.h"
#include "utils/common.h"

using namespace MxRec;
using namespace std;

namespace MxRec {

extern "C" {
drvError_t halHostRegister(void* srcPtr, UINT64 size, UINT32 flag, UINT32 devid, void** dstPtr);
drvError_t halHostUnregister(void* srcPtr, UINT32 devid);
drvError_t rtDeviceGetBareTgid(uint32_t* pid);
}

constexpr uint64_t RMA_SHM_TOTAL_MEM_SIZE = 1 * 1024 * 1024 * 1024 * 20L; // shared memory total size(B)
constexpr int RMA_SHM_QUEUE_CAPACITY = 50;                           // max depth
constexpr int32_t MAX_RANK_SIZE = 4095;
constexpr uint32_t KEY_DIM = 0;
constexpr uint32_t EMB_DIM = 1;

uint32_t g_pid = 0;
bool g_aclInit[MAX_RANK_SIZE] = {false};
std::unordered_map<std::string, void *> g_shmSvmMap;
std::unordered_map<std::string, void *> g_shmAddr;
std::unordered_map<std::string, int> g_shmId;

RmaDevModel g_rmaDevModel = RmaDevModel::PCIE_TH_DEV;

void InitShmHeader(RmaShmHeader* header, int64_t memSize, int32_t capacity)
{
    header->totalMemSize = memSize - RMA_SHM_HEAD_LEN;
    header->queueCapacity = capacity;
    header->seqIn = 0;
    header->seqOut = 0;
    header->frontOffset = RMA_SHM_HEAD_LEN;
    header->tailOffset = RMA_SHM_HEAD_LEN;
    header->buffLimit = 0;
    header->seqOutPre = 0;
    header->frontOffsetPre = RMA_SHM_HEAD_LEN;
}

void ResetShmHeader(RmaShmHeader* header)
{
    header->seqIn = 0;
    header->seqOut = 0;
    header->frontOffset = RMA_SHM_HEAD_LEN;
    header->tailOffset = RMA_SHM_HEAD_LEN;
    header->buffLimit = 0;
    header->seqOutPre = 0;
    header->frontOffsetPre = RMA_SHM_HEAD_LEN;
}

void RmaFreeShm(std::string shmName, void* memory)
{
    if (g_rmaDevModel == RmaDevModel::SVM_MAP_DEV) {
        if (aclrtFreeHost(memory) != ACL_ERROR_NONE) {
            LOG_WARN("Free host mem failed.");
        }
    }

    if (g_rmaDevModel == RmaDevModel::PCIE_TH_DEV) {
        int shmId = g_shmId[shmName];
        (void)shmdt(memory);
        shmctl(shmId, IPC_RMID, nullptr);
        g_shmId.erase(shmName);
        LOG_INFO("Free shm with shmid: {} success.", shmId);
    }
}

bool IsPrefix(const std::string& str, const std::string& prefix)
{
    if (prefix.length() > str.length()) {
        return false;
    }
    return str.compare(0, prefix.length(), prefix) == 0;
}

uint32_t GetRegisterFlag(RmaDevModel mode)
{
    switch (mode) {
        case RmaDevModel::SVM_MAP_DEV:
            return HOST_SVM_MAP_DEV;
        default:
            return HOST_MEM_MAP_DEV_PCIE_TH;
    }
}

void* ShmMemSet(std::string& shmName, uint64_t memSize)
{
    void* memory = nullptr;
    if (g_rmaDevModel == RmaDevModel::SVM_MAP_DEV) {
        if (aclrtMallocHost((void **)&memory, memSize) != ACL_ERROR_NONE) {
            auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                               StringFormat("Malloc host memory failed."));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString());
        }
        (void)aclrtMemset(memory, memSize, 0, memSize);
        LOG_INFO("Create memory {}, size: {} bytes successfully.", shmName.c_str(), memSize);
    } else {
        struct shmid_ds buf;
        key_t key = IPC_PRIVATE;    // create new shared memory every time
        int shmId = -1;
        if (GlobalEnv::hugeTlbEnable) {
            shmId = shmget(key, memSize, IPC_CREAT | SHM_WR_ALL | SHM_HUGETLB);
        } else {
            shmId = shmget(key, memSize, IPC_CREAT | SHM_WR_OWN);
        }
        if (shmId == -1) {
            auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                               StringFormat("Shmget failed."));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString());
        }

        memory = shmat(shmId, nullptr, 0);
        if (memory == reinterpret_cast<void *>(-1)) {
            shmctl(shmId, IPC_RMID, nullptr);
            auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                               StringFormat("Shmat failed."));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString());
        }

        shmctl(shmId, IPC_STAT, &buf);
        (void)memset_s(memory, memSize, 0, memSize);
        g_shmId.insert(std::make_pair(shmName, shmId));
        LOG_INFO("Create shm {}, shmid: {}, size: {} bytes successfully.", shmName.c_str(), shmId, memSize);
    }
    return memory;
}

// malloc shared memory
void *RmaCreateShm(std::string& shmName, uint64_t memSize, int deviceId, int capacity)
{
    string chipName = GetChipName(deviceId);
    if (IsPrefix(chipName, "910B")) {
        g_rmaDevModel = RmaDevModel::PCIE_TH_DEV;
    } else if (IsPrefix(chipName, "910_93")) {
        g_rmaDevModel = RmaDevModel::SVM_MAP_DEV;
    } else {
        auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                           StringFormat("Unsupported chip type: %s.", chipName.c_str()));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString());
    }

    void* memory = ShmMemSet(shmName, memSize);
    uint32_t flag = GetRegisterFlag(g_rmaDevModel);
    void *svmMem = nullptr;
    if (halHostRegister(memory, memSize, flag, deviceId, &svmMem) != DRV_ERROR_NONE) {
        RmaFreeShm(shmName, memory);
        auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                           StringFormat("Device %d halHostRegister failed.", deviceId));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString());
    }
    g_shmAddr.insert(std::make_pair(shmName, memory));
    InitShmHeader(reinterpret_cast<RmaShmHeader *>(memory), memSize, capacity);

    return svmMem;
}

int64_t GetShmAddr(std::string& name, int deviceId, int capacity)
{
    auto memSize = RMA_SHM_TOTAL_MEM_SIZE;
    if (capacity > RMA_SHM_QUEUE_CAPACITY) {
        capacity = RMA_SHM_QUEUE_CAPACITY;
    }

    if (!g_aclInit[deviceId]) {
        aclError retOk = aclInit(nullptr);
        if (retOk != ACL_SUCCESS) {
            LOG_ERROR("aclInit failed, rank:{}", deviceId);
            return false;
        }
        auto ret = aclrtSetDevice(deviceId);
        if (ret != ACL_ERROR_NONE) {
            LOG_ERROR("aclrtSetDevice failed, rank:{}", deviceId);
            return false;
        }
        g_aclInit[deviceId] = true;
    }

    if (g_pid == 0) {
        (void)rtDeviceGetBareTgid(&g_pid);
    }
    std::string shmName = name + "_" + std::to_string(g_pid);
    LOG_INFO("Device {} start alloc shm for {}.", deviceId, shmName.c_str());

    if (g_shmSvmMap.find(shmName) == g_shmSvmMap.end()) {
        g_shmSvmMap.insert(std::make_pair(shmName, RmaCreateShm(shmName, memSize, deviceId, capacity)));
    }

    return reinterpret_cast<int64_t>(g_shmSvmMap[shmName]);
}

// get shm's ddr address for send and recive
void *GetHostAddr(std::string& name)
{
    std::string shmName = name + "_" + std::to_string(g_pid);
    if (g_shmAddr.find(shmName) == g_shmAddr.end()) {
        return nullptr;
    }
    return g_shmAddr[shmName];
}

void FreeShmAddr(uint32_t deviceId)
{
    if (!g_aclInit[deviceId]) {
        return;
    }
    for (auto &pair : g_shmAddr) {
        halHostUnregister(pair.second, deviceId);
        RmaFreeShm(pair.first, pair.second);
        pair.second = nullptr;
        LOG_INFO("Device {} free memory: {} success.", deviceId, pair.first.c_str());
    }
    g_shmAddr.clear();
    g_shmId.clear();
    g_shmSvmMap.clear();
    if (g_aclInit[deviceId]) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        g_aclInit[deviceId] = false;
        LOG_INFO("Device {} free acl device.", deviceId);
    }
}

uint64_t GetShmSeq(RmaShmHeader* queueHeader)
{
    uint64_t sequence = queueHeader->seqIn;
    return ++sequence;
}

void ClearShmQueue()
{
    for (auto &pair : g_shmAddr) {
        ResetShmHeader(reinterpret_cast<RmaShmHeader *>(pair.second));
        LOG_INFO("Reset queue: {}", pair.first);
    }
}

bool FullQueue(RmaShmHeader* queHeader, uint64_t dataSize)
{
    dataSize += RMA_SHM_DATA_HEAD;
    if (queHeader->seqIn - queHeader->seqOut >= queHeader->queueCapacity) {
        return true;
    }
    if (queHeader->tailOffset + dataSize > queHeader->totalMemSize &&
            dataSize + RMA_SHM_HEAD_LEN > queHeader->frontOffset) {
        return true;
    } else {
        if (queHeader->tailOffset < queHeader->frontOffset &&
            queHeader->tailOffset + dataSize >= queHeader->frontOffset) {
            return true;
        }
        if (queHeader->tailOffset == queHeader->frontOffset && queHeader->buffLimit != 0) {
            return true;
        }
    }
    return false;
}

uint8_t *ShmEnqueueHeadRaw(RmaShmHeader* header, std::array<int64_t, RMA_DIM_MAX>& dims, uint64_t sequence)
{
    int64_t dataSize = dims[KEY_DIM] * dims[EMB_DIM] * sizeof(float) * 1L;
    uint8_t *lastPos = nullptr;
    RmaShmData dataHead;
    LOG_INFO("Before enqueue, capacity: {}, seq-in: {}, seq-out: {}.",
             header->queueCapacity, header->seqIn, header->seqOut);
    int loopCnt = 0;
    while (FullQueue(header, dataSize)) {
        this_thread::sleep_for(1ms);
        ++loopCnt;
        if (loopCnt >= MAX_WAIT_LOOP) {
            auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                               StringFormat("check full loop failed"));
            LOG_ERROR(error.ToString());
            throw std::runtime_error(error.ToString());
        }
    }
    dataHead.totalLen = dataSize + RMA_SHM_DATA_HEAD;
    dataHead.dataType = 0;
    dataHead.dimNum = RMA_DIM_MAX;
    dataHead.dataLen = dataSize;
    dataHead.sequence = sequence;
    dataHead.dims[0] = dims[0];
    dataHead.dims[1] = dims[1];
    dataHead.readyLen = 0;

    if (header->tailOffset + dataSize > header->totalMemSize) {
        lastPos = reinterpret_cast<uint8_t *>(header) + RMA_SHM_HEAD_LEN;
        if (memcpy_s(lastPos, RMA_SHM_DATA_HEAD, &dataHead, RMA_SHM_DATA_HEAD) != EOK) {
            auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                               StringFormat("Data head memcpy failed."));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString());
        }
        header->buffLimit = header->tailOffset;
        header->tailOffset = RMA_SHM_HEAD_LEN + dataHead.totalLen; // Offset from the head of the queue
    } else {
        lastPos = reinterpret_cast<uint8_t *>(header) + header->tailOffset;
        if (memcpy_s(lastPos, RMA_SHM_DATA_HEAD, &dataHead, RMA_SHM_DATA_HEAD) != EOK) {
            auto error = Error(ModuleName::M_RMA_SHM_SVM, ErrorType::UNKNOWN,
                               StringFormat("Data head memcpy failed."));
            LOG_ERROR(error.ToString());
            throw runtime_error(error.ToString());
        }
        header->tailOffset += dataHead.totalLen;
    }
    header->seqIn = sequence;
    LOG_INFO("After enqueue, capacity: {}, seq-in: {}, seq-out: {}.",
             header->queueCapacity, header->seqIn, header->seqOut);
    return lastPos;
}

int64_t GetShmElemNum(RmaShmHeader* header)
{
    int64_t queueNum = header->seqIn - header->seqOut;
    return queueNum;
}

RmaShmData *ShmDequeuePre(RmaShmHeader* queHeader)
{
    if (queHeader->seqIn - queHeader->seqOutPre <= 0) {
        return nullptr;
    }

    LOG_DEBUG("Before pre-dequeue, seq-in: {}, seq-out: {}, front: {}, tail: {}, buff-limit: {}.",
              queHeader->seqIn, queHeader->seqOutPre,
              queHeader->frontOffsetPre, queHeader->tailOffset, queHeader->buffLimit);

    if (queHeader->frontOffsetPre == queHeader->buffLimit) {
        queHeader->frontOffsetPre = RMA_SHM_HEAD_LEN;
    }
    RmaShmData *dataHeader = reinterpret_cast<RmaShmData *>(
            reinterpret_cast<uint8_t *>(queHeader) + queHeader->frontOffsetPre);
    uint64_t dataLen = dataHeader->totalLen;
    queHeader->frontOffsetPre += dataLen;
    queHeader->seqOutPre = dataHeader->sequence;
    LOG_DEBUG("After pre-dequeue, data-len: {}, seq-in: {}, seq-out: {}, front: {}, tail: {}, buff-limit: {}.",
              dataLen, queHeader->seqIn, queHeader->seqOutPre,
              queHeader->frontOffsetPre, queHeader->tailOffset, queHeader->buffLimit);
    return dataHeader;
}

RmaShmData *ShmDequeue(RmaShmHeader* queHeader)
{
    if (GetShmElemNum(queHeader) <= 0) {
        return nullptr;
    }
    LOG_DEBUG("Before dequeue, seq-in: {}, seq-out: {}, front: {}, tail: {}, buff-limit: {}.",
              queHeader->seqIn, queHeader->seqOut,
              queHeader->frontOffset, queHeader->tailOffset, queHeader->buffLimit);

    if (queHeader->frontOffset == queHeader->buffLimit) {
        queHeader->frontOffset = RMA_SHM_HEAD_LEN;
        queHeader->buffLimit = 0;
    }
    RmaShmData *dataHeader = reinterpret_cast<RmaShmData *>(
            reinterpret_cast<uint8_t *>(queHeader) + queHeader->frontOffset);

    uint64_t dataLen = dataHeader->totalLen;
    queHeader->frontOffset += dataLen;
    queHeader->seqOut = dataHeader->sequence;
    LOG_DEBUG("After dequeue, data-len: {}, seq-in: {}, seq-out: {}, front: {}, tail: {}, buff-limit: {}.",
              dataLen, queHeader->seqIn, queHeader->seqOut,
              queHeader->frontOffset, queHeader->tailOffset, queHeader->buffLimit);
    return dataHeader;
}

RmaShmData *MallocFromShm(std::string& channelName, std::array<int64_t, RMA_DIM_MAX>& dims)
{
    RmaShmHeader *queueHeader = reinterpret_cast<RmaShmHeader *>(GetHostAddr(channelName));
    if (queueHeader == nullptr) {
        auto error = Error(ModuleName::M_HYBRID_MGMT, ErrorType::INVALID_ARGUMENT,
                           StringFormat("Failed to find valid shm for channel: %s.",
                                        channelName.c_str()));
        LOG_ERROR(error.ToString());
        throw runtime_error(error.ToString());
    }
    auto seq = GetShmSeq(queueHeader);
    RmaShmData *dataHeader = reinterpret_cast<RmaShmData *>(ShmEnqueueHeadRaw(queueHeader, dims, seq));
    return dataHeader;
}

uint8_t *GetDataAddr(RmaShmData* dataHeader)
{
    return reinterpret_cast<uint8_t *>(dataHeader) + RMA_SHM_DATA_HEAD;
}

void SetReadyLen(RmaShmData* dataHeader, uint64_t value)
{
    uint64_t *readyLen = reinterpret_cast<uint64_t *>(reinterpret_cast<uint8_t *>(dataHeader) +
            sizeof(RmaShmData) - sizeof(uint64_t));
    *readyLen = value;
}
}