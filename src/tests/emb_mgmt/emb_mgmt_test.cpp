/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: emb mgmt test
 * Author: MindX SDK
 * Create: 2022
 * History: NA
 */

#include <gtest/gtest.h>
#include "hybrid_mgmt/hybrid_mgmt.h"
#include "host_emb/host_emb.h"
#include "utils/common.h"

using namespace std;
using namespace MxRec;

constexpr int DDR_DEVICE_SIZE = 5;
constexpr int DDR_HOST_SIZE = 15;
constexpr int HBM_DEVICE_SIZE = 20;
constexpr int HBM_HOST_SIZE = 0;

// test class key_process
class EmbMgmtTest : public testing::Test {
protected:
    // create RankInfo rankInfo and const vector<EmbInfo> &embInfos
    RankInfo allRank;
    int rankId = 0;
    int deviceId = 0;
    int rankSize = 1;
    int localRankSize = 1;
    bool useStatic = false;
    std::vector<int> maxStep = { 10, 5 };
    vector<EmbInfo> embInfos;
    EmbInfo embInfo;
    string name = "model";
    int sendCount = 5;
    int embeddingSize = 8;
    int extEmbeddingSize = 24;
    bool isSave = true;
    size_t devVocabSize = 5;
    size_t hostVocabSize = 15;
    vector<RandomInfo> randomInfos;
    RandomInfo randomInfo;
    int start = 0;
    int len = hostVocabSize * embeddingSize;
    float constantVal = 0;
    float randomMin = -0.1f;
    float randomMax = 0.1f;
    int seed = 10086;
    string loadPath;
    HybridMgmt* hybridMgmt;
    vector<InitializeInfo> initializeInfos;
    InitializeInfo initializeInfo;
    ConstantInitializerInfo constantInitializerInfo;
    string constantInitializerName = "constant_initializer";
    int nBatch = 10;

    void UpdateEmb(vector<size_t> &missingKeysHostPos, int channelId, const string &embName,
        std::unique_ptr<HostEmb> &hostEmb, vector<Tensor> &d2h_emb)
    {
        LOG_INFO(HD + "update emb start");
        if (d2h_emb.size() == 0) {
            LOG_INFO(HD + "emb is none channelId:{}", channelId);
            return;
        }

        auto tensorPtr = d2h_emb[0].flat<float>().data();
        for (size_t i = 0; i < missingKeysHostPos.size(); i++) {
            (hostEmb->GetEmb(embName).embData[missingKeysHostPos[i]]).assign(
                tensorPtr,
                tensorPtr + hostEmb->GetEmb(embName).hostEmbInfo.extEmbeddingSize);
            tensorPtr = tensorPtr + hostEmb->GetEmb(embName).hostEmbInfo.extEmbeddingSize;
        }
        for (size_t i = 0; i < hostEmb->GetEmb(embName).embData.size(); ++i) {
            LOG_INFO("hostEmb: embName {}, {} is: {}", embName, i,
                VectorToString(hostEmb->GetEmb(embName).embData[i]));
        }
        LOG_INFO(HD + "update emb end");
        d2h_emb.clear();
    }

    bool Float2TensorVec(const vector<vector<float>>& Datas, vector<Tensor>& tensors)
    {
        tensors.clear();
        for (auto transferData: Datas) {
            Tensor tmpTensor(tensorflow::DT_FLOAT, { (int)transferData.size() });
            auto tmpData = tmpTensor.flat<float>();
            for (uint j = 0; j < transferData.size(); ++j) {
                tmpData(j) = transferData[j];
            }
            tensors.emplace_back(move(tmpTensor));
        }
        return true;
    }

    void SetUp()
    {
        // init key_process (RankInfo rankInfo, const vector<EmbInfo> &embInfos)
        constantInitializerInfo = ConstantInitializerInfo(constantVal, 1);
        initializeInfo = InitializeInfo(constantInitializerName, start, embeddingSize, constantInitializerInfo);
        initializeInfos.push_back(initializeInfo);

        randomInfo = RandomInfo(start, len, constantVal, randomMin, randomMax);
        randomInfos.emplace_back(randomInfo);
    }

    void TearDown()
    {
        // delete
    }
};
#ifndef GTEST
TEST_F(EmbMgmtTest, Initialize)
{
    vector<size_t> vocabsize = { devVocabSize, hostVocabSize };
    aoto param = EmbInfoParams(name, sendCount, embeddingSize, extEmbeddingSize, isSave)
    embInfo = EmbInfo(param, vocabsize, initializeInfos);
    embInfos.emplace_back(embInfo);
    vector<ThresholdValue> thresholdValues = {};

    auto hybridMgmt = Singleton<HybridMgmt>::GetInstance();
    cout << "setup..." << endl;

    allRank = RankInfo(GlogConfig::gRankId, deviceId, localRankSize, useStatic, nBatch, maxStep);
    hybridMgmt->Initialize(allRank, embInfos, seed, thresholdValues, false);
    auto hostEmbs = make_unique<HostEmb>();
    hostEmbs->Initialize(embInfos, seed);
    auto hostHashMaps = make_unique<EmbHashMap>();
    hostHashMaps->Init(allRank, embInfos, false);

    int currentBatchId = 0;
    vector<emb_key_t> lookupKeys = { 1, 3, 5, 7 };
    vector<Tensor> d2h_emb;
    vector<vector<float>> tmpDatas;
    vector<Tensor> tmpData;
    hostHashMaps->Process(embInfo.name, lookupKeys, currentBatchId, tmpData);
    auto missingKeys = hostHashMaps->embHashMaps[embInfo.name].missingKeysHostPos;
    LOG_INFO("missingKeys {}", missingKeys);
    hostEmbs->EmbDataGenerator(initializeInfos, seed, missingKeys.size(), embeddingSize, tmpDatas);
    auto status = Float2TensorVec(tmpDatas, d2h_emb);
    ASSERT_EQ(status, true);
    UpdateEmb(missingKeys, 0, embInfo.name, hostEmbs, d2h_emb);
    hostHashMaps->embHashMaps[embInfo.name].missingKeysHostPos.clear();

    lookupKeys = { 2, 3, 5, 6 };
    hostHashMaps->Process(embInfo.name, lookupKeys, currentBatchId, tmpData);
    missingKeys = hostHashMaps->embHashMaps[embInfo.name].missingKeysHostPos;
    LOG_INFO("missingKeys {}", missingKeys);
    hostEmbs->EmbDataGenerator(initializeInfos, seed, missingKeys.size(), embeddingSize, tmpDatas);
    status = Float2TensorVec(tmpDatas, d2h_emb);
    ASSERT_EQ(status, true);
    UpdateEmb(missingKeys, 0, embInfo.name, hostEmbs, d2h_emb);
    hostHashMaps->embHashMaps[embInfo.name].missingKeysHostPos.clear();

    lookupKeys = { 1, 7, 9, 10 };
    hostHashMaps->Process(embInfo.name, lookupKeys, currentBatchId, tmpData);
    missingKeys = hostHashMaps->embHashMaps[embInfo.name].missingKeysHostPos;
    LOG_INFO("missingKeys {}", missingKeys);
    hostEmbs->EmbDataGenerator(initializeInfos, seed, missingKeys.size(), embeddingSize, tmpDatas);
    Float2TensorVec(tmpDatas, d2h_emb);
    status = Float2TensorVec(tmpDatas, d2h_emb);
    ASSERT_EQ(status, true);
    UpdateEmb(missingKeys, 0, embInfo.name, hostEmbs, d2h_emb);
    hostHashMaps->embHashMaps[embInfo.name].missingKeysHostPos.clear();

    hybridMgmt->Destroy();
}
#endif

#ifndef GTEST
TEST_F(EmbMgmtTest, Initialize_HBM)
{
    devVocabSize = HBM_DEVICE_SIZE;
    hostVocabSize = HBM_HOST_SIZE;
    vector<size_t> vocabsize = { devVocabSize, hostVocabSize };
    aoto param = EmbInfoParams(name, sendCount, embeddingSize, extEmbeddingSize, isSave)
    embInfo = EmbInfo(params, vocabsize, initializeInfos);
    embInfos.emplace_back(embInfo);
    vector<ThresholdValue> thresholdValues;
    thresholdValues.emplace_back(name, 1, 1, 1, true);

    auto hybridMgmt = Singleton<HybridMgmt>::GetInstance();
    cout << "setup..." << endl;
    allRank = RankInfo(GlogConfig::gRankId, deviceId, localRankSize, useStatic, nBatch, maxStep);
    hybridMgmt->Initialize(allRank, embInfos, seed, thresholdValues, false);

    hybridMgmt->Destroy();
}
#endif

#ifndef GTEST
TEST_F(EmbMgmtTest, Evict)
{
    size_t devVocabSize = DDR_DEVICE_SIZE;
    size_t hostVocabSize = DDR_HOST_SIZE;
    vector<size_t> vocabsize = { devVocabSize, hostVocabSize };
    aoto param = EmbInfoParams(name, sendCount, embeddingSize, extEmbeddingSize, isSave)
    embInfo = EmbInfo(params, vocabsize, initializeInfos);
    embInfos.emplace_back(embInfo);
    vector<ThresholdValue> thresholdValues;
    thresholdValues.emplace_back(name, 1, 1, 1, true);

    auto hybridMgmt = Singleton<HybridMgmt>::GetInstance();
    cout << "setup..." << endl;
    allRank = RankInfo(GlogConfig::gRankId, deviceId, localRankSize, true, nBatch, maxStep);
    hybridMgmt->Initialize(allRank, embInfos, seed, thresholdValues, false);

    // evict test, ddr
    hybridMgmt->Evict();

    hybridMgmt->Destroy();
}
#endif

TEST_F(EmbMgmtTest, Evict_HBM)
{
#ifndef GTEST
    devVocabSize = HBM_DEVICE_SIZE;
    hostVocabSize = HBM_HOST_SIZE;
    vector<size_t> vocabsize = { devVocabSize, hostVocabSize };
    aoto param = EmbInfoParams(name, sendCount, embeddingSize, extEmbeddingSize, isSave)
    embInfo = EmbInfo(params, vocabsize, initializeInfos);
    embInfos.emplace_back(embInfo);
    vector<ThresholdValue> thresholdValues;
    thresholdValues.emplace_back(name, 1, 1, 1, true);

    auto hybridMgmt = Singleton<HybridMgmt>::GetInstance();
    cout << "setup..." << endl;
    allRank = RankInfo(GlogConfig::gRankId, deviceId, localRankSize, true, nBatch, maxStep);
    hybridMgmt->Initialize(allRank, embInfos, seed, thresholdValues, false);

    // evict test, hbm
    vector<emb_key_t> keys = { 1, 3, 5, 7 };
    hybridMgmt->EvictKeys(name, keys);

    hybridMgmt->Destroy();
#endif
}
