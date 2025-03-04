/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>
#include <ctime>
#include <random>
#include <iostream>
#include <memory>
#include "utils/common.h"
#include "hybrid_mgmt/hybrid_mgmt.h"
#include "hd_transfer/rma_shm_svm.h"

using namespace std;
using namespace MxRec;
using namespace testing;

class RmaShmSvmTest :  : public testing::Test {
protected:
    int deviceId = 0;
    int capacity = 5;
    std::string name = "test";
    std::string sendName = "test";
    int64_t dims[2] = {10, 128};

    std::vector<float> CreateBatchData(){
        std::uniform_real_distribution<float> u(-1, 1);
        std::default_random_engine e(time(NULL));
        std::vector<float> ret;
        for(int64_t i = 0; i < dims[0]; i++){
            for(int64_t j = 0; j < dims[1]; j++){
                ret.push_back(u(e));
            }
        }
        return ret;
    }

    void PushBatch() {
        std::vector<float> data = CreteBatchData();
        auto dataHeader = MallocFromShm(sendName, dims);
        float* h2dEmb = reinterpret_cast<float*>(GetDataAddr(dataHeader));
        uint64_t memSize = dims[1] * sizeof(float);
        for(uint64_t i = 0; i < data.size(); i++){
            auto rc = memcpy_s(h2dEmb + i, sizeof(float), &(data[i]), sizeof(float));
        }
        SetReadyLen(dataHeader, dims[0] * memSize);
        return;
    }
};

TEST_F(RmaShmSvmTest, GetShmAddr)
{
    int64_t ret = GetShmAddr(name, deviceId, capacity);
    ASSERT_NE(ret, 0);
}

TEST_F(RmaShmSvmTest, EnqueueAndDequeue)
{
    PushBatch();
    RmaShmHeader* queueHeader = reinterpret_cast<RmaShmHeader*>(GetHostAddr(name));
    int64_t queueNum = GetShmElemNum(queueHeader);
    ASSERT_EQ(queueNum, 1);

    PushBatch();
    PushBatch();
    PushBatch();
    PushBatch();
    queueHeader = reinterpret_cast<RmaShmHeader*>(GetHostAddr(name));
    queueNum = GetShmElemNum(queueHeader);
    ASSERT_EQ(queueNum, 5);

    RmaShmData* dataHead = ShmDequeuePre(queueHeader);
    ASSERT_NE(dataHead, nullptr);
    queueNum = GetShmElemNum(queueHeader);
    ASSERT_EQ(queueNum, 5);
    dataHead = ShmDequeue(queueHeader);
    ASSERT_NE(dataHead, nullptr);
    queueNum = GetShmElemNum(queueHeader);
    ASSERT_EQ(queueNum, 4);
    PushBatch();
    queueNum = GetShmElemNum(queueHeader);
    ASSERT_EQ(queueNum, 5);
}

TEST_F(RmaShmSvmTest, FreeShmAddr)
{
    FreeShmAddr(deviceId);
    auto ptr = GetHostAddr(name);
    ASSERT_EQ(ptr, nullptr);
}
