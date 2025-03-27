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

#include "rma_swap_test.h"

#include <random>
#include <thread>

#include "utils/logger.h"

using namespace std;

namespace TfTest {
#define OMP_THREADS 16
#define PRECISION_TEST false
constexpr size_t TIMEOUT = 300000;

int64_t GetShmElemNum(MxRec::RmaShmHeader* header)
{
    int64_t queueNum = header->seqIn - header->seqOut;
    return queueNum;
}

// 仅用于pybind侧调用，read共享内存；
int ReadShmData(std::string name, bool save, std::string savePath)
{
    auto* queueHead = reinterpret_cast<MxRec::RmaShmHeader*>(MxRec::GetHostAddr(name));;
    if (queueHead == nullptr) {
        LOG_ERROR("shm addr is invalid");
        return 0;
    }

    size_t timeOutCount = 0;
    while (GetShmElemNum(queueHead) <= 0) {
        this_thread::sleep_for(1ms);
        if (timeOutCount++ > TIMEOUT) {
            LOG_WARN("shm read time out, queue is empty.");
            return 0;
        }
    }

    MxRec::RmaShmData* dataHead = MxRec::ShmDequeuePre((MxRec::RmaShmHeader*)queueHead);
    if (dataHead == nullptr) {
        LOG_ERROR("shm outqueue failed");
        return 0;
    }

    float* dataAddr = (float*)MxRec::GetDataAddr(dataHead);

    uint64_t dataLen = dataHead->dataLen;
    LOG_DEBUG("queue data-len: %lld", dataLen);
    if (save) {
        DumpVector(dataAddr, dataLen / sizeof(float),
                   savePath + "/" + name + "_" + std::to_string(dataHead->sequence) + ".bin");
    }

    MxRec::ShmDequeue((MxRec::RmaShmHeader*)queueHead);
    return 1;
}

// 仅用于pybind侧调用，write共享内存；
int WriteShmData(std::string name, std::vector <int64_t>& shape, bool save, std::string savePath)
{
    std::array<int64_t, RMA_DIM_MAX> dims= {shape[0], shape[1]};

    MxRec::RmaShmData* dataHead = MxRec::MallocFromShm(name, dims);
    if (dataHead == nullptr) {
        LOG_ERROR("malloc failed");
        return 0;
    }

    int64_t dataLen = dataHead->dataLen;
    int64_t totalLen = dataHead->totalLen;
    LOG_DEBUG("write shm %s, total-len %lld, data-len %lld", name.c_str(), totalLen, dataLen);

    float* dataAddr = (float*)MxRec::GetDataAddr(dataHead);
#if PRECISION_TEST
    std::default_random_engine e;
    e.seed(std::random_device{}());
    std::uniform_real_distribution<float> distribution(-1, 1);
#pragma omp parallel for num_threads(OMP_THREADS) default(none) shared(dataLen, dataAddr, distribution, e)
    for (int i = 0; i < (dataLen / sizeof(float)); i++) {
        dataAddr[i] = distribution(e);
    }
#endif
    if (save) {
        DumpVector(dataAddr, dataLen / sizeof(float),
                   savePath + "/" + name + "_" + std::to_string(dataHead->sequence) + ".bin");
    }

    MxRec::SetReadyLen(dataHead, dataLen);
    return 1;
}

}