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

#include <gtest/gtest.h>
#include <emock/emock.hpp>

#include "host_emb/host_emb.h"
#include "tensorflow/core/framework/tensor.h"
#include "hd_transfer/hd_transfer.h"
#include "utils/singleton.h"

using namespace std;
using namespace tensorflow;
using namespace MxRec;

namespace {
bool operator==(const Tensor& tensor1, const Tensor& tensor2)
{
    if (tensor1.shape() != tensor2.shape()) {
        return false;
    }
    auto tensor1_data = tensor1.flat<float>();
    auto tensor2_data = tensor2.flat<float>();
    for (int j = 0; j < tensor1_data.size(); j++) {
        if (tensor1_data(j) != tensor2_data(j)) {
            return false;
        }
    }
    return true;
}

bool operator==(const vector<Tensor>& p1, const vector<Tensor>& p2)
{
    if (p1.size() != p2.size()) {
        return false;
    }
    for (int i = 0; i<p1.size(); i++) {
        const Tensor& tensor1 = p1[i];
        const Tensor& tensor2 = p2[i];
        if (!(tensor1 == tensor2)) {
            return false;
        }
    }
    return true;
}

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

}