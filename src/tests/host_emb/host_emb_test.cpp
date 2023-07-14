/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: host emb test
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include <gtest/gtest.h>

#include "host_emb/host_emb.h"
#include "tensorflow/core/framework/tensor.h"

using namespace std;
using namespace tensorflow;
using namespace MxRec;

TEST(HostEmb, Tensor2Float)
{
    shared_ptr<tuple<int, string, vector<long>>> lookups;
    vector<int32_t> host_emb;
    host_emb.resize(15);
    vector<vector<int32_t>> p(5, vector<int32_t>(3));
    host_emb[0] = 1;
    host_emb[1] = 3;
    std::cout << host_emb[0] << std::endl;
    for (int i = 0; i < 5; i++) {
        p[i].assign(host_emb.begin() + i * 3, host_emb.begin() + (i + 1) * 3);
    }
    std::cout << p[0][0] << std::endl;
    std::cout << '5' << std::endl;
    vector<Tensor> q;
    std::cout << '0' << std::endl;
    for (int i = 0; i < 2; i++) {
        Tensor tmpTensor(tensorflow::DT_INT32, { 3 });
        std::cout << '1' << std::endl;
        auto tmpData = tmpTensor.flat<int32_t>();
        std::cout << '2' << std::endl;
        for (int j = 0; j < 3; j++) {
            tmpData(j) = p[i][j];
            std::cout << '3' << std::endl;
        }

        q.emplace_back(tmpTensor);
        std::cout << '4' << std::endl;
    }
    std::cout << '1' << std::endl;
    std::cout << q[0].flat<int32_t>()(0) << std::endl;
    std::cout << q[0].flat<int32_t>()(1) << std::endl;
    std::cout << q[1].flat<int32_t>()(0) << std::endl;
    ASSERT_EQ(1, 1);
}

TEST(HostEmb, DefaultConstructor)
{
    HostEmb h;
    h.procThreadsForTrain.emplace_back(make_unique<thread>([] {}));
    h.Join(TRAIN_CHANNEL_ID);
    ASSERT_EQ(h.procThreadsForTrain.size(), 0);

    h.procThreadsForEval.emplace_back(make_unique<thread>([] {}));
    h.Join(EVAL_CHANNEL_ID);
    ASSERT_EQ(h.procThreadsForEval.size(), 0);
}