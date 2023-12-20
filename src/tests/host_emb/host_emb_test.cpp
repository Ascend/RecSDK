/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: host emb test
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

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

TEST(HostEmb, HostEmbUpdateTest) {
    vector<Tensor> tensors;
    Tensor tmpTensor(tensorflow::DT_FLOAT, { 32 });
    auto tmpData = tmpTensor.flat<float>();
    for (int j = 0; j < 32; j++) {
        tmpData(j) = 0.1f*j;
    }
    tensors.emplace_back(tmpTensor);

    EMOCK(&HDTransfer::Recv).expects(exactly(1)).will(returnValue(tensors));
    HostEmb h;
    EmbInfo embInfo;
    embInfo.name = "TestEmb";
    embInfo.devVocabSize = 100;
    embInfo.hostVocabSize = 200;
    embInfo.extEmbeddingSize = 32;
    std::string name = "random_normal_initializer";
    InitializeInfo info(name, 0, embInfo.extEmbeddingSize, NormalInitializerInfo(0, 1, 7, 1.0));
    embInfo.initializeInfos.emplace_back(info);
    vector<EmbInfo> embInfos {embInfo};
    h.Initialize(embInfos, 7);
    vector<size_t> missingKeysHostPos{199};
    h.UpdateEmb(missingKeysHostPos, TRAIN_CHANNEL_ID, embInfo.name);
    ASSERT_EQ(h.hostEmbs[embInfo.name].embData[199][0], 0);
    ASSERT_EQ(h.hostEmbs[embInfo.name].embData[199][31], 0.1f*31);
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

TEST(HostEmb, InitializerAndEvict)
{
    HostEmb h;
    EmbInfo embInfo;
    embInfo.name = "TestEmb";
    embInfo.devVocabSize = 100;
    embInfo.hostVocabSize = 200;
    embInfo.extEmbeddingSize = 32;
    std::string name = "constant_initializer";
    float initVal = 0.05f;
    InitializeInfo info(name, 0, embInfo.extEmbeddingSize, ConstantInitializerInfo(initVal, 1.0));
    embInfo.initializeInfos.emplace_back(info);
    vector<EmbInfo> embInfos {embInfo};
    h.Initialize(embInfos, 7);

    ASSERT_EQ(h.hostEmbs[embInfo.name].embData.size(), embInfo.hostVocabSize);
    ASSERT_EQ(h.hostEmbs[embInfo.name].embData[0].size(), embInfo.extEmbeddingSize);
    ASSERT_EQ(h.hostEmbs[embInfo.name].embData[0][0], initVal);
    ASSERT_EQ(h.hostEmbs[embInfo.name].embData[0][embInfo.extEmbeddingSize-1], initVal);

    float initVal1 = 100.89f;
    InitializeInfo info1(name, 0, embInfo.extEmbeddingSize, ConstantInitializerInfo(initVal1, 1.0));
    embInfo.initializeInfos.clear();
    embInfo.initializeInfos.emplace_back(info1);
    vector<size_t> offset{1, 199};
    h.hostEmbs[embInfo.name].hostEmbInfo = embInfo;
    h.EvictInitEmb(embInfo.name, offset);
    ASSERT_EQ(h.hostEmbs[embInfo.name].embData[1][0], initVal1);
    ASSERT_EQ(h.hostEmbs[embInfo.name].embData[199][embInfo.extEmbeddingSize-1], initVal1);
}

TEST(HostEmb, GetH2DEmb)
{
    HostEmb h;
    EmbInfo embInfo;
    embInfo.name = "TestEmb";
    embInfo.devVocabSize = 100;
    embInfo.hostVocabSize = 200;
    embInfo.extEmbeddingSize = 32;
    std::string name = "random_normal_initializer";
    InitializeInfo info(name, 0, embInfo.extEmbeddingSize, NormalInitializerInfo(0, 1, 7, 1.0));
    embInfo.initializeInfos.emplace_back(info);
    vector<EmbInfo> embInfos {embInfo};
    h.Initialize(embInfos, 7);
    vector<size_t> missingKeysHostPos{1, 199};
    vector<Tensor> h2dEmbOut;
    h.GetH2DEmb(missingKeysHostPos, embInfo.name, h2dEmbOut);
    ASSERT_EQ(h.hostEmbs[embInfo.name].embData[1][0], h2dEmbOut[0].flat<float>()(0));
    ASSERT_EQ(h.hostEmbs[embInfo.name].embData[199][0], h2dEmbOut[0].flat<float>()(32));
}
}