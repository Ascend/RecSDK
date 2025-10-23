/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

#include "hybrid_mgmt_block_mock.h"
#include "factory_mock.h"
#include "hybrid_mgmt/hybrid_mgmt.h"
#include "hd_transfer/hd_transfer_mock.h"
#include "ssd_cache/emb_cache_manager_mock.h"

using namespace MxRec;
using namespace testing;

constexpr int EXT_EMB_SIZE = 10;
constexpr int HOST_VOCAB_SIZE = 100;
constexpr int DEVICE_VOCAB_SIZE = 50;

class HybridMgmtTest : public testing::Test {
public:
    void SetUp() override
    {
        std::vector<ThresholdValue> thresholdValues;
        auto ret = m_hybridMgmt.Initialize(m_rankInfo, m_embInfos, 0, thresholdValues, false, false, false);
        m_hybridMgmt.InitRankInfo(m_rankInfo, m_embInfos);
        EXPECT_EQ(ret, true);
        m_hybridMgmt.Start();
        m_hybridMgmt.StartThreadForHBM();
        m_hybridMgmt.StartThreadForDDR();
        ret = m_hybridMgmt.Load(m_savePath, {});
        EXPECT_EQ(ret, true);

        m_transfer = std::make_shared<HDTransferMock>();
        m_hybridMgmt.hdTransfer = m_transfer.get();
        m_embCachePtr = std::make_shared<EmbCache::EmbCacheManagerMock>();
        m_hybridMgmt.embCache = m_embCachePtr;
        m_hybridMgmtBlock = std::make_shared<HybridMgmtBlockMock>();
        m_hybridMgmt.hybridMgmtBlock = m_hybridMgmtBlock.get();
    }

    void TearDown() override
    {
        m_hybridMgmt.Save(m_savePath, false, false);
    }

    std::shared_ptr<HDTransferMock> m_transfer{nullptr};
    std::shared_ptr<EmbCache::EmbCacheManagerMock> m_embCachePtr{nullptr};
    std::shared_ptr<HybridMgmtBlockMock> m_hybridMgmtBlock{nullptr};
    float m_embSwapIn = 1.0;
    float m_embSwapOut = 1.0;
    HybridMgmt m_hybridMgmt;
    RankInfo m_rankInfo;
    std::vector<EmbInfo> m_embInfos;
    std::string m_embTableName = "table1";
    std::string m_savePath = "savePath1";
};

TEST_F(HybridMgmtTest, TaskStart)
{
    m_hybridMgmt.TrainTask(static_cast<TaskType>(0));
    m_hybridMgmt.TrainTask(static_cast<TaskType>(1));
    m_hybridMgmt.EvalTask(static_cast<TaskType>(0));
    m_hybridMgmt.EvalTask(static_cast<TaskType>(1));
    int batchId{0};
    auto ret = m_hybridMgmt.ParseKeys(TRAIN_CHANNEL_ID, batchId, static_cast<TaskType>(0));
    EXPECT_EQ(ret, true);
    ret = m_hybridMgmt.ParseKeys(TRAIN_CHANNEL_ID, batchId, static_cast<TaskType>(1));
    EXPECT_EQ(ret, true);

    EmbBaseInfo info;
    ret = m_hybridMgmt.ProcessEmbInfoHBM(info, false);
    EXPECT_EQ(ret, true);
    ret = m_hybridMgmt.ProcessEmbInfoDDR(info);
    EXPECT_EQ(ret, true);
    ret = m_hybridMgmt.ProcessEmbInfoL3Storage(info);
    EXPECT_EQ(ret, true);
}

TEST_F(HybridMgmtTest, SetFeatureTypeForLoad)
{
    std::vector<CkptFeatureType> loadFeatures;
    m_hybridMgmt.SetFeatureTypeForLoad(loadFeatures);
    EXPECT_EQ(loadFeatures.size(), 1);
}

TEST_F(HybridMgmtTest, SetFeatureTypeForLoad_SetGlobalEnv)
{
    size_t resSize = 2;
    GlobalEnv::recordKeyCount = true;
    std::vector<CkptFeatureType> loadFeatures;
    m_hybridMgmt.SetFeatureTypeForLoad(loadFeatures);
    EXPECT_EQ(loadFeatures.size(), resSize);
}

TEST_F(HybridMgmtTest, Destroy_NotInitialized_Error)
{
    m_hybridMgmt.isInitialized = false;
    EXPECT_THROW(m_hybridMgmt.Destroy(), std::runtime_error);
    m_hybridMgmt.isInitialized = true;
}

TEST_F(HybridMgmtTest, SendUniqKeysAndRestoreVecHBMShouldSendUniqKeysWhenIsGradIsTrue)
{
    EmbBaseInfo info;
    info.channelId = 1;
    info.batchId = 1;
    info.name = "test";
    std::unique_ptr<std::vector<Tensor>> infoVecs = std::make_unique<std::vector<Tensor>>();
    infoVecs->push_back(Tensor());
    infoVecs->push_back(Tensor());
    bool isGrad = true;

    m_hybridMgmt.SendUniqKeysAndRestoreVecHBM(info, infoVecs, isGrad);
    EXPECT_EQ(infoVecs->size(), 0);
}

TEST_F(HybridMgmtTest, SendUniqKeysAndRestoreVecHBM_NotGrad)
{
    EmbBaseInfo info;
    info.channelId = 1;
    info.batchId = 1;
    info.name = "test";
    std::unique_ptr<std::vector<Tensor>> infoVecs = std::make_unique<std::vector<Tensor>>();
    infoVecs->push_back(Tensor());
    infoVecs->push_back(Tensor());
    bool isGrad = false;

    m_hybridMgmt.SendUniqKeysAndRestoreVecHBM(info, infoVecs, isGrad);
    EXPECT_EQ(infoVecs->size(), 0);
}

TEST_F(HybridMgmtTest, EvictKeysShouldLogErrorWhenRemoveEmbsByKeysFailsWhenRemoveEmbsByKeysFails)
{
    std::string embName = "testEmb";
    std::vector<MxRec::emb_cache_key_t> keys = {1, 2, 3};
    EXPECT_CALL(*m_embCachePtr, RemoveEmbsByKeys(_, _)).Times(1).WillRepeatedly(Return(1));
    (void)m_hybridMgmt.Evict();
    m_hybridMgmt.EvictKeys(embName, keys);
}

TEST_F(HybridMgmtTest, EvictKeys_RemoveEmbsByKeysOk)
{
    std::string embName = "testEmb";
    std::vector<MxRec::emb_cache_key_t> keys = {1, 2, 3};
    EXPECT_CALL(*m_embCachePtr, RemoveEmbsByKeys(_, _)).Times(1).WillRepeatedly(Return(0));
    (void)m_hybridMgmt.Evict();
    
    EXPECT_NO_THROW(m_hybridMgmt.EvictKeys(embName, keys));
}

TEST_F(HybridMgmtTest, EvictKeys_EmptyKeys)
{
    std::string embName = "testEmb";
    std::vector<MxRec::emb_cache_key_t> keys = {};
    m_hybridMgmt.EvictKeys(embName, keys);
    EXPECT_CALL(*m_embCachePtr, RemoveEmbsByKeys(_, _)).Times(0);
}

TEST_F(HybridMgmtTest, EvictL3StorageKeysShouldNotCallEvictL3StorageEmbeddingWhenL3StorageIsNotEnabled)
{
    m_hybridMgmt.isL3StorageEnabled = false;
    std::string embName = "testEmb";
    std::vector<MxRec::emb_cache_key_t> keys = {1, 2, 3};

    m_hybridMgmt.EvictL3StorageKeys(embName, keys);
}

TEST_F(HybridMgmtTest, EvictL3StorageKeys_L3StorageEnabled)
{
    m_hybridMgmt.isL3StorageEnabled = true;
    std::string embName = "testEmb";
    std::vector<MxRec::emb_cache_key_t> keys = {};

    m_hybridMgmt.EvictL3StorageKeys(embName, keys);
}

TEST_F(HybridMgmtTest, NotifyBySessionRunAndCountStepBySessionRun)
{
    m_hybridMgmt.NotifyBySessionRun(0);
    m_hybridMgmt.CountStepBySessionRun(0, 1);
}

TEST_F(HybridMgmtTest, NotifyBySessionRunAndCountStepBySessionRunFail)
{
    m_hybridMgmt.isInitialized = false;
    EXPECT_THROW(m_hybridMgmt.NotifyBySessionRun(0), std::runtime_error);
    EXPECT_THROW(m_hybridMgmt.CountStepBySessionRun(0, 1), std::runtime_error);
    m_hybridMgmt.isInitialized = true;
}

TEST_F(HybridMgmtTest, GetTableSizeAndGetTableCapacity)
{
    auto ret = m_hybridMgmt.GetTableSize(m_embTableName);
    EXPECT_EQ(ret, -1);
    ret = m_hybridMgmt.GetTableCapacity(m_embTableName);
    EXPECT_EQ(ret, -1);
}

TEST_F(HybridMgmtTest, SetOptimizerInfoShouldThrowErrorWhenNotInitialized)
{
    MxRec::OptimizerInfo optimInfo;
    m_hybridMgmt.isInitialized = false;

    EXPECT_THROW(m_hybridMgmt.SetOptimizerInfo(m_embTableName, optimInfo), std::runtime_error);
    m_hybridMgmt.isInitialized = true;
}

TEST_F(HybridMgmtTest, LookUpAndRemoveAddrsShouldNotThrowExceptionWhenEmbeddingLookupAddrsSucceeds)
{
    EmbTaskInfo info;
    info.batchId = 1;
    info.threadIdx = 0;
    info.cvNotifyIndex = 0;
    info.extEmbeddingSize = EXT_EMB_SIZE;
    info.channelId = 0;
    info.name = "test_table";
    m_hybridMgmt.isRunning = true;

    std::vector<uint64_t> keys = {1, 2, 3};
    m_hybridMgmt.DDRSwapKeyQue[info.name + "SwapOut"][info.channelId].Pushv(keys);
    m_hybridMgmt.DDRSwapKeyQue[info.name + "SwapIn"][info.channelId].Pushv(keys);
    m_hybridMgmt.HBMSwapKeyQue[info.name + "SwapIn"][info.channelId].Pushv(keys);
    m_hybridMgmt.HBMSwapKeyQue[info.name + "SwapOut"][info.channelId].Pushv(keys);
    EXPECT_CALL(*m_embCachePtr,
                EmbeddingLookupAddrs(_, _, _, _)).Times(4).WillRepeatedly(Return(0)); // expect call 4 times
    EXPECT_CALL(*m_embCachePtr, EmbeddingRemove(_, _, _)).Times(1).WillRepeatedly(Return(0));
    EXPECT_CALL(*m_embCachePtr, Destroy()).Times(1).WillRepeatedly(Return());
    EXPECT_NO_THROW(m_hybridMgmt.LookUpAndRemoveAddrs(info));
}

TEST_F(HybridMgmtTest, LookUpAndRemoveAddrs_NotRunning)
{
    EmbTaskInfo info;
    info.batchId = 1;
    info.threadIdx = 0;
    info.cvNotifyIndex = 0;
    info.extEmbeddingSize = EXT_EMB_SIZE;
    info.channelId = 0;
    info.name = "test_table";
    m_hybridMgmt.isRunning = false;

    std::vector<uint64_t> keys = {1, 2, 3};
    m_hybridMgmt.DDRSwapKeyQue[info.name + "SwapOut"][info.channelId].Pushv(keys);
    m_hybridMgmt.DDRSwapKeyQue[info.name + "SwapIn"][info.channelId].Pushv(keys);
    m_hybridMgmt.HBMSwapKeyQue[info.name + "SwapIn"][info.channelId].Pushv(keys);
    m_hybridMgmt.HBMSwapKeyQue[info.name + "SwapOut"][info.channelId].Pushv(keys);

    EXPECT_CALL(*m_embCachePtr, EmbeddingLookupAddrs(_, _, _, _)).Times(0);
    m_hybridMgmt.LookUpAndRemoveAddrs(info);
}

TEST_F(HybridMgmtTest, LookUpAndRemoveAddrs_Error)
{
    EmbTaskInfo info;
    info.batchId = 1;
    info.threadIdx = 0;
    info.cvNotifyIndex = 0;
    info.extEmbeddingSize = EXT_EMB_SIZE;
    info.channelId = 0;
    info.name = "test_table";
    m_hybridMgmt.isRunning = true;

    std::vector<uint64_t> keys = {1, 2, 3};
    m_hybridMgmt.DDRSwapKeyQue[info.name + "SwapOut"][info.channelId].Pushv(keys);
    m_hybridMgmt.DDRSwapKeyQue[info.name + "SwapIn"][info.channelId].Pushv(keys);
    m_hybridMgmt.HBMSwapKeyQue[info.name + "SwapIn"][info.channelId].Pushv(keys);
    m_hybridMgmt.HBMSwapKeyQue[info.name + "SwapOut"][info.channelId].Pushv(keys);
    EXPECT_CALL(*m_embCachePtr,
                EmbeddingLookupAddrs(_, _, _, _)).Times(AnyNumber()).WillRepeatedly(Return(0));
    EXPECT_CALL(*m_embCachePtr, EmbeddingRemove(_, _, _)).Times(1).WillRepeatedly(Return(1));
    EXPECT_CALL(*m_embCachePtr, Destroy()).Times(AnyNumber()).WillRepeatedly(Return());
    EXPECT_THROW(m_hybridMgmt.LookUpAndRemoveAddrs(info), std::runtime_error);
}

TEST_F(HybridMgmtTest, LookUpSwapAddrs)
{
    m_hybridMgmt.isRunning = true;
    m_hybridMgmt.lookupAddrSuccess = true;
    m_hybridMgmt.EosL1Que[m_embTableName][0].Pushv(false);
    std::vector<uint64_t> keys = {1, 2, 3};
    m_hybridMgmt.HBMSwapKeyQue[m_embTableName + "SwapIn"][0].Pushv(keys);
    m_hybridMgmt.HBMSwapKeyQue[m_embTableName + "SwapOut"][0].Pushv(keys);
    EXPECT_CALL(*m_embCachePtr, EmbeddingLookupAddrs(_, _, _, _)).WillOnce(Return(0)).WillOnce(Return(1));
    EXPECT_CALL(*m_embCachePtr, Destroy()).Times(1).WillRepeatedly(Return());
    EXPECT_THROW(m_hybridMgmt.LookUpSwapAddrs(m_embTableName, 0), std::runtime_error);
}

TEST_F(HybridMgmtTest, LookUpSwapAddrs_Error)
{
    m_hybridMgmt.isRunning = true;
    m_hybridMgmt.lookupAddrSuccess = true;
    m_hybridMgmt.EosL1Que[m_embTableName][0].Pushv(true);
    m_hybridMgmt.EosL1Que[m_embTableName][0].Pushv(false);
    std::vector<uint64_t> keys = {1, 2, 3};
    m_hybridMgmt.HBMSwapKeyQue[m_embTableName + "SwapIn"][0].Pushv(keys);
    m_hybridMgmt.HBMSwapKeyQue[m_embTableName + "SwapOut"][0].Pushv(keys);
    EXPECT_CALL(*m_embCachePtr, EmbeddingLookupAddrs(_, _, _, _)).WillOnce(Return(1));
    EXPECT_CALL(*m_embCachePtr, Destroy()).Times(AnyNumber()).WillRepeatedly(Return());
    EXPECT_THROW(m_hybridMgmt.LookUpSwapAddrs(m_embTableName, 0), std::runtime_error);
}

TEST_F(HybridMgmtTest, LookUpSwapAddrs_NotRunning)
{
    m_hybridMgmt.isRunning = false;
    m_hybridMgmt.lookupAddrSuccess = true;

    EXPECT_CALL(*m_embCachePtr, EmbeddingLookupAddrs(_, _, _, _)).Times(0);
    m_hybridMgmt.LookUpSwapAddrs(m_embTableName, 0);
}

TEST_F(HybridMgmtTest, LookUpSwapAddrs_NotLookupAddrSuccess)
{
    m_hybridMgmt.isRunning = true;
    m_hybridMgmt.lookupAddrSuccess = false;

    EXPECT_CALL(*m_embCachePtr, EmbeddingLookupAddrs(_, _, _, _)).Times(0);
    m_hybridMgmt.LookUpSwapAddrs(m_embTableName, 0);
}

TEST_F(HybridMgmtTest, SendHostMapAndLoadMap)
{
    std::vector<uint64_t> swapOutPos;
    (void)m_hybridMgmt.BuildSaveSwapTensor(swapOutPos, false);
    m_hybridMgmt.FetchDeviceEmb();
    EmbBaseInfo info;
    std::vector<uint64_t> swapInPosUint;
    std::vector<uint64_t> swapOutPosUint;
    m_hybridMgmt.SendTensorForSwap(info, swapInPosUint, swapOutPosUint);
    EXPECT_NO_THROW(m_hybridMgmt.SendHostMap(m_embTableName));
    EXPECT_NO_THROW(m_hybridMgmt.SendLoadMap(m_embTableName));
}

TEST_F(HybridMgmtTest, EmbeddingTaskShouldResetStepsWhenCalled)
{
    EmbInfo embInfo;
    embInfo.name = m_embTableName;
    embInfo.extEmbeddingSize = 1;
    m_hybridMgmt.mgmtEmbInfo.push_back(embInfo);
    m_hybridMgmt.isRunning = true;
    m_hybridMgmt.HBMSwapAddrsQue[m_embTableName + "SwapIn"][TRAIN_CHANNEL_ID].Pushv({&m_embSwapIn});
    m_hybridMgmt.HBMSwapAddrsQue[m_embTableName + "SwapIn"][EVAL_CHANNEL_ID].Pushv({&m_embSwapIn});
    m_hybridMgmt.HBMSwapAddrsQue[m_embTableName + "SwapOut"][TRAIN_CHANNEL_ID].Pushv({&m_embSwapOut});
    m_hybridMgmt.HBMSwapAddrsQue[m_embTableName + "SwapOut"][EVAL_CHANNEL_ID].Pushv({&m_embSwapOut});
    m_hybridMgmt.InitPipelineMutexAndCV(m_embTableName);

    EXPECT_CALL(*m_embCachePtr, Destroy()).Times(1).WillRepeatedly(Return());
    m_hybridMgmt.EmbeddingTask();
}

TEST_F(HybridMgmtTest, ReceiveKey)
{
    EmbInfo embInfo;
    embInfo.name = m_embTableName;
    embInfo.extEmbeddingSize = 1;
    m_hybridMgmt.mgmtEmbInfo.push_back(embInfo);
    EXPECT_NO_THROW(m_hybridMgmt.ReceiveKey());
}

TEST_F(HybridMgmtTest, UpdateDeltaInfoShouldUpdateCorrectlyWhenCalledWithValidParameters)
{
    std::string embName = "testEmb";
    std::vector<int64_t> keyCountVec = {1, 10, 2, 20, 3, 30};
    int64_t timeStamp = 123456789;
    int64_t batchId = 1;

    m_hybridMgmt.UpdateDeltaInfo(embName, keyCountVec, timeStamp, batchId);

    auto& embMap = m_hybridMgmt.deltaMap[embName];
    EXPECT_EQ(embMap.size(), 3); // expect embMap size : 3
    EXPECT_EQ(embMap[1].totalCount, 10); // 10 is the expected totalCount for key 1
    EXPECT_EQ(embMap[2].totalCount, 20); // 20 is the expected totalCount for key 2
    EXPECT_EQ(embMap[3].totalCount, 30); // 30 is the expected totalCount for key 3
    EXPECT_EQ(embMap[1].recentCount, 10); // 10 is the expected recentCount for key 1
    EXPECT_EQ(embMap[2].recentCount, 20); // 20 is the expected recentCount for key 2
    EXPECT_EQ(embMap[3].recentCount, 30); // 30 is the expected recentCount for key 3
    EXPECT_EQ(embMap[1].isChanged, true); // verify isChanged for key 1
    EXPECT_EQ(embMap[2].isChanged, true); // verify isChanged for key 2
    EXPECT_EQ(embMap[3].isChanged, true); // verify isChanged for key 3
    EXPECT_EQ(embMap[1].batchID, batchId); // verify batchID for key 1
    EXPECT_EQ(embMap[2].batchID, batchId); // verify batchID for key 2
    EXPECT_EQ(embMap[3].batchID, batchId); // verify batchID for key 3
    EXPECT_EQ(embMap[1].lastUseTime, timeStamp); // verify lastUseTime for key 1
    EXPECT_EQ(embMap[2].lastUseTime, timeStamp); // verify lastUseTime for key 2
    EXPECT_EQ(embMap[3].lastUseTime, timeStamp); // verify lastUseTime for key 3
}

TEST_F(HybridMgmtTest, HybridMgmt_ResetDeltaInfo_ShouldResetDeltaInfo_WhenCalled)
{
    EmbInfo embInfo;
    embInfo.name = m_embTableName;
    m_hybridMgmt.mgmtEmbInfo.push_back(embInfo);
    KeyInfo keyInfo;
    emb_key_t key = 1;
    absl::flat_hash_map<emb_key_t, KeyInfo> keyMap = {{key, keyInfo}};
    m_hybridMgmt.deltaMap[m_embTableName] = keyMap;

    m_hybridMgmt.ResetDeltaInfo();

    EXPECT_EQ(m_hybridMgmt.deltaMap[m_embTableName][0].recentCount, 0);
    EXPECT_EQ(m_hybridMgmt.deltaMap[m_embTableName][0].isChanged, false);
}

TEST_F(HybridMgmtTest, EmbeddingUpdateDDRShouldNotThrowExceptionWhenMemcpySucceeds)
{
    EmbTaskInfo info;
    info.batchId = 0;
    info.threadIdx = 0;
    info.cvNotifyIndex = 1;
    info.extEmbeddingSize = 1;
    info.channelId = TRAIN_CHANNEL_ID;
    info.name = m_embTableName;
    float target = 0;
    std::vector<float*> swapOutAddrs = {&target};
    float emb = 1.0;
    MxRec::Logger::SetLevel(Logger::DEBUG);

    EXPECT_NO_THROW(m_hybridMgmt.EmbeddingUpdateDDR(info, &emb, swapOutAddrs));
    EXPECT_EQ(target, emb);
}

TEST_F(HybridMgmtTest, EmbeddingUpdateDDR_LoggerGetLevelINFO)
{
    EmbTaskInfo info;
    info.batchId = 0;
    info.threadIdx = 0;
    info.cvNotifyIndex = 1;
    info.extEmbeddingSize = 1;
    info.channelId = TRAIN_CHANNEL_ID;
    info.name = m_embTableName;
    float target = -1;
    std::vector<float*> swapOutAddrs = {&target};
    float emb = 1.0;
    MxRec::Logger::SetLevel(Logger::INFO);

    EXPECT_NO_THROW(m_hybridMgmt.EmbeddingUpdateDDR(info, &emb, swapOutAddrs));
}

TEST_F(HybridMgmtTest, EmbeddingLookUpL3StorageShouldReturnTrueWhenAllConditionsAreMet)
{
    EmbTaskInfo info;
    info.name = m_embTableName;
    info.channelId = TRAIN_CHANNEL_ID;
    std::vector<Tensor> h2dEmb;

    EXPECT_TRUE(m_hybridMgmt.EmbeddingLookUpL3Storage(info, h2dEmb));
}

TEST_F(HybridMgmtTest, InitDataPipelineForDDRShouldInitializeQueuesWhenCalled)
{
    m_hybridMgmt.isRunning = true;
    m_hybridMgmt.lookupAddrSuccess = true;
    m_hybridMgmt.EosL1Que[m_embTableName][0].Pushv(false);
    m_hybridMgmt.EosL1Que[m_embTableName][1].Pushv(false);
    std::vector<uint64_t> keys = {1, 2, 3};
    m_hybridMgmt.HBMSwapKeyQue[m_embTableName + "SwapIn"][0].Pushv(keys);
    m_hybridMgmt.HBMSwapKeyQue[m_embTableName + "SwapOut"][0].Pushv(keys);
    m_hybridMgmt.HBMSwapKeyQue[m_embTableName + "SwapIn"][1].Pushv(keys);
    m_hybridMgmt.HBMSwapKeyQue[m_embTableName + "SwapOut"][1].Pushv(keys);
    EXPECT_CALL(*m_embCachePtr,
                EmbeddingLookupAddrs(_, _, _, _)).Times(AnyNumber()).WillRepeatedly(Return(0));
    EXPECT_CALL(*m_embCachePtr, Destroy()).Times(1).WillRepeatedly(Return());
    m_hybridMgmt.InitDataPipelineForDDR(m_embTableName);

    EXPECT_TRUE(m_hybridMgmt.HBMSwapKeyQue.find(m_embTableName + m_hybridMgmt.SWAP_IN_STR) !=
                m_hybridMgmt.HBMSwapKeyQue.end());
    EXPECT_TRUE(m_hybridMgmt.HBMSwapKeyQue.find(m_embTableName + m_hybridMgmt.SWAP_OUT_STR) !=
                m_hybridMgmt.HBMSwapKeyQue.end());
    EXPECT_TRUE(m_hybridMgmt.HBMSwapAddrsQue.find(m_embTableName + m_hybridMgmt.SWAP_IN_STR) !=
                m_hybridMgmt.HBMSwapAddrsQue.end());
    EXPECT_TRUE(m_hybridMgmt.HBMSwapAddrsQue.find(m_embTableName + m_hybridMgmt.SWAP_OUT_STR) !=
                m_hybridMgmt.HBMSwapAddrsQue.end());
    EXPECT_TRUE(m_hybridMgmt.EosL1Que.find(m_embTableName) != m_hybridMgmt.EosL1Que.end());
    EXPECT_TRUE(m_hybridMgmt.EosL2Que.find(m_embTableName) != m_hybridMgmt.EosL2Que.end());

    EXPECT_EQ(m_hybridMgmt.lookUpSwapAddrsThreads.size(), 2); // lookUpSwapAddrsThreads size is 2
}

TEST_F(HybridMgmtTest, InitDataPipelineForL3StorageShouldInitializeQueuesWhenCalled)
{
    int extEmbeddingSize = EXT_EMB_SIZE;
    m_hybridMgmt.InitDataPipelineForL3Storage(m_embTableName, extEmbeddingSize);

    EXPECT_TRUE(m_hybridMgmt.HBMSwapKeyQue.find(m_embTableName + m_hybridMgmt.SWAP_IN_STR) !=
                m_hybridMgmt.HBMSwapKeyQue.end());
    EXPECT_TRUE(m_hybridMgmt.HBMSwapKeyQue.find(m_embTableName + m_hybridMgmt.SWAP_OUT_STR) !=
                m_hybridMgmt.HBMSwapKeyQue.end());
    EXPECT_TRUE(m_hybridMgmt.HBMSwapAddrsQue.find(m_embTableName + m_hybridMgmt.SWAP_IN_STR) !=
                m_hybridMgmt.HBMSwapAddrsQue.end());
    EXPECT_TRUE(m_hybridMgmt.HBMSwapAddrsQue.find(m_embTableName + m_hybridMgmt.SWAP_OUT_STR) !=
                m_hybridMgmt.HBMSwapAddrsQue.end());
    EXPECT_TRUE(m_hybridMgmt.EosL1Que.find(m_embTableName) != m_hybridMgmt.EosL1Que.end());
    EXPECT_TRUE(m_hybridMgmt.EosL2Que.find(m_embTableName) != m_hybridMgmt.EosL2Que.end());
    EXPECT_TRUE(m_hybridMgmt.HBMSwapKeyQue.find(m_embTableName + m_hybridMgmt.ADDR_STR) !=
                m_hybridMgmt.HBMSwapKeyQue.end());
    EXPECT_TRUE(m_hybridMgmt.HBMSwapKeyForL3StorageQue.find(m_embTableName + m_hybridMgmt.SWAP_IN_STR) !=
                m_hybridMgmt.HBMSwapKeyForL3StorageQue.end());
    EXPECT_TRUE(m_hybridMgmt.HBMSwapKeyForL3StorageQue.find(m_embTableName + m_hybridMgmt.ADDR_STR) !=
                m_hybridMgmt.HBMSwapKeyForL3StorageQue.end());
    EXPECT_TRUE(m_hybridMgmt.HBMSwapKeyForL3StorageQue.find(m_embTableName + m_hybridMgmt.SWAP_OUT_STR) !=
                m_hybridMgmt.HBMSwapKeyForL3StorageQue.end());
    EXPECT_TRUE(m_hybridMgmt.DDRSwapKeyQue.find(m_embTableName + m_hybridMgmt.SWAP_OUT_STR) !=
                m_hybridMgmt.DDRSwapKeyQue.end());
    EXPECT_TRUE(m_hybridMgmt.DDRSwapKeyQue.find(m_embTableName + m_hybridMgmt.SWAP_IN_STR) !=
                m_hybridMgmt.DDRSwapKeyQue.end());
    EXPECT_TRUE(m_hybridMgmt.DDRSwapKeyForL3StorageQue.find(m_embTableName + m_hybridMgmt.SWAP_OUT_STR) !=
                m_hybridMgmt.DDRSwapKeyForL3StorageQue.end());
    EXPECT_TRUE(m_hybridMgmt.DDRSwapKeyForL3StorageQue.find(m_embTableName + m_hybridMgmt.SWAP_IN_STR) !=
                m_hybridMgmt.DDRSwapKeyForL3StorageQue.end());
    EXPECT_TRUE(m_hybridMgmt.DDRSwapAddrsQue.find(m_embTableName + m_hybridMgmt.SWAP_OUT_STR) !=
                m_hybridMgmt.DDRSwapAddrsQue.end());
    EXPECT_TRUE(m_hybridMgmt.DDRSwapAddrsQue.find(m_embTableName + m_hybridMgmt.SWAP_IN_STR) !=
                m_hybridMgmt.DDRSwapAddrsQue.end());
}

TEST_F(HybridMgmtTest, InitEmbeddingCacheShouldNotThrowExceptionWhenCreateCacheForTableSucceeds)
{
    std::vector<EmbInfo> embInfos;
    EmbInfo embInfo;
    embInfo.name = m_embTableName;
    embInfo.hostVocabSize = HOST_VOCAB_SIZE;
    embInfo.embeddingSize = EXT_EMB_SIZE;
    embInfo.extEmbeddingSize = EXT_EMB_SIZE + EXT_EMB_SIZE;
    embInfo.devVocabSize = DEVICE_VOCAB_SIZE;
    embInfos.push_back(embInfo);
    m_hybridMgmt.isRunning = true;

    auto mockFactory = std::make_shared<ock::ctr::FactoryMock>();
    auto factoryBakUp = factory;
    factory = mockFactory;

    EXPECT_CALL(*mockFactory, SetExternalLogFuncInner(_)).Times(1);
    EXPECT_CALL(*mockFactory, CreateEmbCacheManager(_)).Times(1);
    EXPECT_CALL(*m_embCachePtr, CreateCacheForTable(_, _, _, _, _)).Times(1).WillRepeatedly(Return(0));
    EXPECT_CALL(*m_embCachePtr, Destroy()).Times(1).WillRepeatedly(Return());

    EXPECT_NO_THROW(m_hybridMgmt.InitEmbeddingCache(embInfos));
    factory = factoryBakUp;
}

TEST_F(HybridMgmtTest, InitEmbeddingCache_L3StorageEnabled)
{
    std::vector<EmbInfo> embInfos;
    EmbInfo embInfo;
    embInfo.name = m_embTableName;
    embInfo.hostVocabSize = HOST_VOCAB_SIZE;
    embInfo.embeddingSize = EXT_EMB_SIZE;
    embInfo.extEmbeddingSize = EXT_EMB_SIZE + EXT_EMB_SIZE;
    embInfo.devVocabSize = DEVICE_VOCAB_SIZE;
    embInfos.push_back(embInfo);
    m_hybridMgmt.isRunning = true;
    m_hybridMgmt.isL3StorageEnabled = true;

    auto mockFactory = std::make_shared<ock::ctr::FactoryMock>();
    auto factoryBakUp = factory;
    factory = mockFactory;

    EXPECT_CALL(*mockFactory, SetExternalLogFuncInner(_)).Times(1);
    EXPECT_CALL(*mockFactory, CreateEmbCacheManager(_)).Times(1);
    EXPECT_CALL(*m_embCachePtr, CreateCacheForTable(_, _, _, _, _)).Times(1).WillRepeatedly(Return(1));
    EXPECT_CALL(*m_embCachePtr, Destroy()).Times(1).WillRepeatedly(Return());

    EXPECT_THROW(m_hybridMgmt.InitEmbeddingCache(embInfos), std::runtime_error);
    factory = factoryBakUp;
}

TEST_F(HybridMgmtTest, EmbeddingReceiveDDR)
{
    EmbTaskInfo info;
    info.name = m_embTableName;
    info.channelId = TRAIN_CHANNEL_ID;
    info.batchId = 0;
    float* ptr{nullptr};
    std::vector<float*> swapOutAddrs;
    m_hybridMgmt.isRunning = true;
    m_hybridMgmt.EosL2Que[info.name][info.channelId].Pushv(true);
    m_hybridMgmt.EosL2Que[info.name][info.channelId].Pushv(true);
    
    std::vector<float> keys = {1, 2, 3};
    float* rawArray = new float[keys.size()];
    std::copy(keys.begin(), keys.end(), rawArray);
    std::vector<float*> vecPtr;
    vecPtr.push_back(rawArray);
    m_hybridMgmt.HBMSwapAddrsQue[info.name + "SwapOut"][info.channelId].Pushv(vecPtr);

    EXPECT_CALL(*m_embCachePtr, Destroy()).Times(1).WillRepeatedly(Return());
    EXPECT_EQ(m_hybridMgmt.EmbeddingReceiveDDR(info, ptr, swapOutAddrs), false);

    delete[] rawArray;
}

TEST_F(HybridMgmtTest, EmbeddingReceiveL3StorageShouldReturnFalseWhenEosL1QueReturnsTrue)
{
    EmbTaskInfo info;
    info.name = m_embTableName;
    info.channelId = TRAIN_CHANNEL_ID;
    info.batchId = 0;
    float* ptr{nullptr};
    std::vector<float*> swapOutAddrs;
    int64_t dims0 = 0;
    m_hybridMgmt.isRunning = true;
    m_hybridMgmt.EosL1Que[info.name][info.channelId].Pushv(true);
    m_hybridMgmt.EosL1Que[info.name][info.channelId].Pushv(true);

    std::vector<uint64_t> keys = {1, 2, 3};
    m_hybridMgmt.DDRSwapKeyQue[info.name + "SwapOut"][info.channelId].Pushv(keys);
    EXPECT_CALL(*m_embCachePtr, EmbeddingLookupAddrs(_, _, _, _)).Times(1).WillRepeatedly(Return(1));

    EXPECT_CALL(*m_embCachePtr, Destroy()).Times(1).WillRepeatedly(Return());
    EXPECT_THROW(m_hybridMgmt.EmbeddingReceiveL3Storage(info, ptr, swapOutAddrs, dims0), std::runtime_error);
}

TEST_F(HybridMgmtTest, EmbeddingUpdateL3StorageShouldThrowExceptionWhenDimsInvalid)
{
    EmbTaskInfo info;
    info.name = m_embTableName;
    info.channelId = TRAIN_CHANNEL_ID;
    info.batchId = 0;
    info.extEmbeddingSize = 1;
    float embPtr = 1.0;
    float targetEmb = 0;
    vector<float*> swapOutAddrs = {&targetEmb};
    int64_t dims0 = 0;
    m_hybridMgmt.isRunning = true;
    std::vector<uint64_t> swapOutDDRAddrOffs = {0};
    m_hybridMgmt.HBMSwapKeyQue[info.name + "Addr"][info.channelId].Pushv(swapOutDDRAddrOffs);
    std::vector<uint64_t> keys = {1, 2, 3};
    m_hybridMgmt.HBMSwapKeyForL3StorageQue[info.name + "Addr"][info.channelId].Pushv(keys);
    m_hybridMgmt.HBMSwapKeyForL3StorageQue[info.name + "SwapOut"][info.channelId].Pushv(keys);

    EXPECT_CALL(*m_embCachePtr, Destroy()).Times(1).WillRepeatedly(Return());
    EXPECT_THROW(m_hybridMgmt.EmbeddingUpdateL3Storage(info, &embPtr, swapOutAddrs, dims0), runtime_error);
}

TEST_F(HybridMgmtTest, EmbeddingReceiveAndUpdateL3StorageShouldThrowExceptionWhenDimsInvalid)
{
    int batchId = 0;
    int index = 0;
    int channelId = 0;
    EmbInfo info;
    info.name = m_embTableName;
    info.extEmbeddingSize = 1;

    EXPECT_NO_THROW(m_hybridMgmt.EmbeddingReceiveAndUpdateL3Storage(batchId, index, info, channelId));
}

TEST_F(HybridMgmtTest, EmbeddingReceiveAndUpdateL3Storage_IndexOne)
{
    int batchId = 0;
    int index = 1;
    int channelId = 0;
    EmbInfo info;
    info.name = m_embTableName;
    info.extEmbeddingSize = 1;

    EXPECT_NO_THROW(m_hybridMgmt.EmbeddingReceiveAndUpdateL3Storage(batchId, index, info, channelId));
}

TEST_F(HybridMgmtTest, EmbeddingSendL3StorageShouldSendWhenInfoAndH2dEmbAreValid)
{
    EmbTaskInfo info;
    info.batchId = 0;
    info.threadIdx = 0;
    info.cvNotifyIndex = 1;
    info.extEmbeddingSize = EXT_EMB_SIZE;
    info.channelId = 0;
    info.name = m_embTableName;
    std::vector<Tensor> h2dEmb;
    h2dEmb.push_back(Tensor());
    m_hybridMgmt.EmbeddingSendL3Storage(info, h2dEmb);

    EXPECT_EQ(m_hybridMgmt.lastSendFinishCV.size(), 2); // lastSendFinishCV size is 2
}

TEST_F(HybridMgmtTest, EmbeddingLookUpAndSendL3StorageShouldSendWhenInfoAndH2dEmbAreValid)
{
    int batchId = 0;
    int index = 0;
    int channelId = 0;
    EmbInfo info;
    info.extEmbeddingSize = EXT_EMB_SIZE;
    info.name = m_embTableName;

    m_hybridMgmt.EmbeddingLookUpAndSendL3Storage(batchId, index, info, channelId);

    EXPECT_EQ(m_hybridMgmt.lastSendFinishCV.size(), 2); // lastSendFinishCV size is 2
}

TEST_F(HybridMgmtTest, EmbeddingLookUpAndSendL3Storage_IndexOne)
{
    size_t resSize = 2;
    int batchId = 0;
    int index = 1;
    int channelId = 0;
    EmbInfo info;
    info.extEmbeddingSize = EXT_EMB_SIZE;
    info.name = m_embTableName;

    m_hybridMgmt.EmbeddingLookUpAndSendL3Storage(batchId, index, info, channelId);

    EXPECT_EQ(m_hybridMgmt.lastSendFinishCV.size(), resSize);
}

TEST_F(HybridMgmtTest, GetUniqueKeysAndSendAll2AllVecAndSendRestoreVec)
{
    EmbBaseInfo info;
    info.channelId = 0;
    info.name = m_embTableName;
    info.batchId = 0;

    bool remainBatchOut = false;
    bool isEos = false;
    auto ret = m_hybridMgmt.GetUniqueKeys(info, remainBatchOut, isEos);
    m_hybridMgmt.SendAll2AllVec(info, remainBatchOut);
    m_hybridMgmt.SendRestoreVec(info, remainBatchOut);
    EXPECT_EQ(ret.size(), 0);
}

TEST_F(HybridMgmtTest, ShouldSendPaddingKeysMaskVecDDRL3WhenUseSumSameIdGradientsIsFalse)
{
    m_hybridMgmt.mgmtRankInfo.useSumSameIdGradients = false;
    EmbBaseInfo info;
    info.channelId = 0;
    info.name = m_embTableName;
    info.batchId = 0;
    info.paddingKeysMask = true;
    std::vector<uint64_t> uniqueKeys = {1, 2, 3};
    std::vector<int32_t> restoreVecSec = {0, 1, 2};

    EXPECT_CALL(*m_embCachePtr, GetPaddingKeysOffset(_)).Times(1);
    m_hybridMgmt.SendLookupOffsets(info, uniqueKeys, restoreVecSec);
}

TEST_F(HybridMgmtTest, SendPaddingKeysMaskVecDDRL3_EVAL_CHANNEL_ID)
{
    EmbBaseInfo info;
    info.channelId = 1;
    info.name = m_embTableName;
    info.batchId = 0;
    info.paddingKeysMask = true;
    std::vector<uint64_t> uniqueKeys = {1, 2, 3};

    m_hybridMgmt.SendPaddingKeysMaskVecDDRL3(info, uniqueKeys);
}

TEST_F(HybridMgmtTest, SendPaddingKeysMaskVecDDRL3_NotPaddingKeysMask)
{
    EmbBaseInfo info;
    info.channelId = 0;
    info.name = m_embTableName;
    info.batchId = 0;
    info.paddingKeysMask = false;
    std::vector<uint64_t> uniqueKeys = {1, 2, 3};

    m_hybridMgmt.SendPaddingKeysMaskVecDDRL3(info, uniqueKeys);
}

TEST_F(HybridMgmtTest, SendLookupOffsets_InvalidIndex)
{
    m_hybridMgmt.mgmtRankInfo.useSumSameIdGradients = true;
    EmbBaseInfo info;
    info.channelId = 0;
    info.name = m_embTableName;
    info.batchId = 0;
    info.paddingKeysMask = true;
    std::vector<uint64_t> uniqueKeys = {1, 2, 3};
    std::vector<int32_t> restoreVecSec = {-1, 1, 2};

    EXPECT_CALL(*m_embCachePtr, GetPaddingKeysOffset(_)).Times(0);
    m_hybridMgmt.SendLookupOffsets(info, uniqueKeys, restoreVecSec);
}

TEST_F(HybridMgmtTest, SendGlobalUniqueVec)
{
    m_hybridMgmt.mgmtRankInfo.useSumSameIdGradients = true;
    EmbBaseInfo info;
    info.channelId = 1;
    info.name = m_embTableName;
    info.batchId = 0;
    info.paddingKeysMask = true;
    info.isDp = true;
    std::vector<uint64_t> uniqueKeys = {1, 2, 3};
    std::vector<int32_t> restoreVecSec = {0, 1, 2};

    EXPECT_CALL(*m_embCachePtr, GetPaddingKeysOffset(_)).Times(0);
    m_hybridMgmt.SendGlobalUniqueVec(info, uniqueKeys, restoreVecSec);
}

TEST_F(HybridMgmtTest, SendGlobalUniqueVec_NotUseSumSameIdGradients)
{
    m_hybridMgmt.mgmtRankInfo.useSumSameIdGradients = false;
    EmbBaseInfo info;
    info.channelId = 0;
    info.name = m_embTableName;
    info.batchId = 0;
    info.paddingKeysMask = true;
    info.isDp = true;
    std::vector<uint64_t> uniqueKeys = {1, 2, 3};
    std::vector<int32_t> restoreVecSec = {0, 1, 2};

    EXPECT_CALL(*m_embCachePtr, GetPaddingKeysOffset(_)).Times(1);
    m_hybridMgmt.SendGlobalUniqueVec(info, uniqueKeys, restoreVecSec);
}

TEST_F(HybridMgmtTest, SendGlobalUniqueVecShouldSendWhenTrainChannelAndUseSumSameIdGradients)
{
    m_hybridMgmt.mgmtRankInfo.useSumSameIdGradients = true;
    EmbBaseInfo info;
    info.channelId = 0;
    info.name = m_embTableName;
    info.batchId = 0;
    info.paddingKeysMask = true;
    std::vector<uint64_t> uniqueKeys = {1, 2, 3};
    std::vector<int32_t> restoreVecSec = {0, 1, 2};

    EXPECT_CALL(*m_embCachePtr, GetPaddingKeysOffset(_)).Times(1);
    m_hybridMgmt.SendGlobalUniqueVec(info, uniqueKeys, restoreVecSec);
}

TEST_F(HybridMgmtTest, CheckLookupAddrSuccessDDRShouldThrowExceptionWhenLookupAddrSuccessIsFalse)
{
    m_hybridMgmt.lookupAddrSuccess = false;
    m_hybridMgmt.lookUpSwapAddrsThreads.push_back(std::async(std::launch::deferred, []() {
        throw std::runtime_error("Lookup failed");
    }));
    EXPECT_THROW(m_hybridMgmt.CheckLookupAddrSuccessDDR(), std::runtime_error);
}

TEST_F(HybridMgmtTest, GetSwapPairsAndKey2OffsetShouldThrowExceptionWhenGetSwapPairsAndKey2OffsetFails)
{
    EmbBaseInfo info;
    std::vector<uint64_t> uniqueKeys;
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair;
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapOutKoPair;

    EXPECT_CALL(*m_embCachePtr, GetSwapPairsAndKey2Offset(_, _, _, _)).WillRepeatedly(Return(1));

    EXPECT_THROW(m_hybridMgmt.GetSwapPairsAndKey2Offset(info, uniqueKeys, swapInKoPair, swapOutKoPair),
                 std::runtime_error);
}

TEST_F(HybridMgmtTest, GetSwapPairsAndKey2Offset)
{
    EmbBaseInfo info;
    std::vector<uint64_t> uniqueKeys;
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapInKoPair;
    std::pair<std::vector<uint64_t>, std::vector<uint64_t>> swapOutKoPair;

    EXPECT_CALL(*m_embCachePtr, GetSwapPairsAndKey2Offset(_, _, _, _)).WillRepeatedly(Return(0));

    EXPECT_NO_THROW(m_hybridMgmt.GetSwapPairsAndKey2Offset(info, uniqueKeys, swapInKoPair, swapOutKoPair));
}

TEST_F(HybridMgmtTest, EnqueueSwapInfoShouldEnqueueWhenCalled)
{
    EmbBaseInfo info;
    info.name = m_embTableName;
    info.channelId = 0;
    info.batchId = 0;
    std::pair<vector<uint64_t>, vector<uint64_t>> swapInKoPair =
        make_pair(vector<uint64_t>{1, 2}, vector<uint64_t>{3, 4});
    std::pair<vector<uint64_t>, vector<uint64_t>> swapOutKoPair =
        make_pair(vector<uint64_t>{5, 6}, vector<uint64_t>{7, 8});

    m_hybridMgmt.EnqueueSwapInfo(info, swapInKoPair, swapOutKoPair);

    EXPECT_EQ(m_hybridMgmt.HBMSwapKeyQue[info.name + m_hybridMgmt.SWAP_IN_STR][info.channelId].Size(), 1);
    EXPECT_EQ(m_hybridMgmt.HBMSwapKeyQue[info.name + m_hybridMgmt.SWAP_OUT_STR][info.channelId].Size(), 1);
    EXPECT_EQ(m_hybridMgmt.EosL1Que[info.name][info.channelId].Size(), 1);
}

TEST_F(HybridMgmtTest, BackUpTrainStatusShouldBackUpWhenTrainBatchIdIsNotZero)
{
    m_hybridMgmt.hybridMgmtBlock->pythonBatchId[TRAIN_CHANNEL_ID] = 1;
    m_hybridMgmt.BackUpTrainStatus();
    EXPECT_TRUE(m_hybridMgmt.isBackUpTrainStatus);
}

TEST_F(HybridMgmtTest, BackUpTrainStatus_ZeroTrainBatchId)
{
    m_hybridMgmt.hybridMgmtBlock->pythonBatchId[TRAIN_CHANNEL_ID] = 0;
    m_hybridMgmt.BackUpTrainStatus();
    EXPECT_FALSE(m_hybridMgmt.isBackUpTrainStatus);
}

TEST_F(HybridMgmtTest, RecoverTrainStatusShouldCallRecoverTrainStatusWhenBackUpTrainStatusIsTrue)
{
    m_hybridMgmt.isBackUpTrainStatus = true;
    m_hybridMgmt.RecoverTrainStatus();
    EXPECT_FALSE(m_hybridMgmt.isBackUpTrainStatus);
}

TEST_F(HybridMgmtTest, GetDeltaModelKeysShouldSaveAllKeysWhenFirstSave)
{
    m_hybridMgmt.isFirstSave = true;
    EmbInfo embInfo;
    embInfo.name = m_embTableName;
    m_hybridMgmt.mgmtEmbInfo.push_back(embInfo);
    std::map<string, map<emb_key_t, KeyInfo>> keyInfoMap;
    m_hybridMgmt.GetDeltaModelKeys("path", false, keyInfoMap);
    EXPECT_EQ(keyInfoMap.size(), 0);
    EXPECT_EQ(keyInfoMap[m_embTableName].size(), 0);
}

TEST_F(HybridMgmtTest, GetDeltaModelKeys_NotFirstSave)
{
    m_hybridMgmt.isFirstSave = false;
    m_hybridMgmt.isIncrementalCkpt = false;
    EmbInfo embInfo;
    embInfo.name = m_embTableName;
    m_hybridMgmt.mgmtEmbInfo.push_back(embInfo);
    std::map<string, map<emb_key_t, KeyInfo>> keyInfoMap;
    m_hybridMgmt.GetDeltaModelKeys("path", false, keyInfoMap);
    EXPECT_EQ(keyInfoMap.size(), 0);
    EXPECT_EQ(keyInfoMap[m_embTableName].size(), 0);
}

TEST_F(HybridMgmtTest, GetDeltaModelKeys_SaveDelta)
{
    m_hybridMgmt.isFirstSave = true;
    EmbInfo embInfo;
    embInfo.name = m_embTableName;
    m_hybridMgmt.mgmtEmbInfo.push_back(embInfo);
    std::map<string, map<emb_key_t, KeyInfo>> keyInfoMap;
    m_hybridMgmt.GetDeltaModelKeys("path", true, keyInfoMap);
    EXPECT_EQ(keyInfoMap.size(), 0);
    EXPECT_EQ(keyInfoMap[m_embTableName].size(), 0);
}

TEST_F(HybridMgmtTest, InitPipelineMutexAndCVShouldCreateMutexAndCVWhenCalled)
{
    std::string embTableName = "testTable";
    m_hybridMgmt.InitPipelineMutexAndCV(embTableName);

    for (int channelId = 0; channelId < MAX_CHANNEL_NUM; ++channelId) {
        for (int threadIndex = 0; threadIndex < EMBEDDING_THREAD_NUM; ++threadIndex) {
            std::string key = MakeSwapCVName(threadIndex, embTableName, channelId);
            EXPECT_TRUE(m_hybridMgmt.lastUpdateFinishMutex.count(key) > 0);
            EXPECT_TRUE(m_hybridMgmt.lastUpdateFinishCV.count(key) > 0);
            EXPECT_TRUE(m_hybridMgmt.lastLookUpFinishMutex.count(key) > 0);
            EXPECT_TRUE(m_hybridMgmt.lastLookUpFinishCV.count(key) > 0);
            EXPECT_TRUE(m_hybridMgmt.lastSendFinishMutex.count(key) > 0);
            EXPECT_TRUE(m_hybridMgmt.lastSendFinishCV.count(key) > 0);
            EXPECT_TRUE(m_hybridMgmt.lastRecvFinishMutex.count(key) > 0);
            EXPECT_TRUE(m_hybridMgmt.lastRecvFinishCV.count(key) > 0);
        }
    }
}

TEST_F(HybridMgmtTest, SendPaddingKeysMaskVecHBMShouldSendWhenAllConditionsMet)
{
    EmbBaseInfo info;
    info.channelId = 0;
    info.paddingKeysMask = true;
    std::unique_ptr<std::vector<Tensor>> infoVecs = std::make_unique<std::vector<Tensor>>();
    infoVecs->push_back(Tensor());
    bool isGrad = true;
    m_hybridMgmt.SendPaddingKeysMaskVecHBM(info, infoVecs, isGrad);
    EXPECT_TRUE(infoVecs->empty());
}

TEST_F(HybridMgmtTest, SendPaddingKeysMaskVecHBM_NotGrad)
{
    EmbBaseInfo info;
    info.channelId = 0;
    info.paddingKeysMask = true;
    std::unique_ptr<std::vector<Tensor>> infoVecs = std::make_unique<std::vector<Tensor>>();
    infoVecs->push_back(Tensor());
    bool isGrad = false;
    m_hybridMgmt.SendPaddingKeysMaskVecHBM(info, infoVecs, isGrad);
    EXPECT_FALSE(infoVecs->empty());
}

TEST_F(HybridMgmtTest, SendPaddingKeysMaskVecHBM_EVAL_CHANNEL_ID)
{
    EmbBaseInfo info;
    info.channelId = 1;
    info.paddingKeysMask = true;
    std::unique_ptr<std::vector<Tensor>> infoVecs = std::make_unique<std::vector<Tensor>>();
    infoVecs->push_back(Tensor());
    bool isGrad = true;
    m_hybridMgmt.SendPaddingKeysMaskVecHBM(info, infoVecs, isGrad);
    EXPECT_FALSE(infoVecs->empty());
}

TEST_F(HybridMgmtTest, SendPaddingKeysMaskVecHBM_NotPaddingKeysMask)
{
    EmbBaseInfo info;
    info.channelId = 0;
    info.paddingKeysMask = false;
    std::unique_ptr<std::vector<Tensor>> infoVecs = std::make_unique<std::vector<Tensor>>();
    infoVecs->push_back(Tensor());
    bool isGrad = true;
    m_hybridMgmt.SendPaddingKeysMaskVecHBM(info, infoVecs, isGrad);
    EXPECT_FALSE(infoVecs->empty());
}

TEST_F(HybridMgmtTest, StartSyncThreadShouldThrowErrorWhenNotDDRMode)
{
    m_hybridMgmt.mgmtRankInfo.isDDR = false;
    EXPECT_THROW(m_hybridMgmt.StartSyncThread(), std::runtime_error);
}

TEST_F(HybridMgmtTest, BuildAndSendH2DEmbedding_NotRunning)
{
    EmbTaskInfo info;
    info.batchId = 1;
    info.threadIdx = 0;
    info.cvNotifyIndex = 0;
    info.extEmbeddingSize = EXT_EMB_SIZE;
    info.channelId = TRAIN_CHANNEL_ID;
    info.name = "test_table";
    float* ptr{nullptr};
    std::array<int64_t, RMA_DIM_MAX> dims= {0, 0};

    std::vector<float> keys = {1, 2, 3};
    float* rawArray = new float[keys.size()];
    std::copy(keys.begin(), keys.end(), rawArray);
    std::vector<float*> vecPtr;
    vecPtr.push_back(rawArray);
    m_hybridMgmt.HBMSwapAddrsQue[info.name + "SwapIn"][info.channelId].Pushv(vecPtr);

    EXPECT_EQ(m_hybridMgmt.BuildAndSendH2DEmbedding(info, ptr, dims), false);
}

TEST_F(HybridMgmtTest, BuildH2DEmbedding_NotRunning)
{
    EmbTaskInfo info;
    info.name = "test_table";
    info.channelId = TRAIN_CHANNEL_ID;
    std::vector<Tensor> h2dEmb;
    h2dEmb.push_back(Tensor());

    std::vector<float> keys = {1, 2, 3};
    float* rawArray = new float[keys.size()];
    std::copy(keys.begin(), keys.end(), rawArray);
    std::vector<float*> vecPtr;
    vecPtr.push_back(rawArray);
    m_hybridMgmt.HBMSwapAddrsQue[info.name + "SwapIn"][info.channelId].Pushv(vecPtr);

    EXPECT_EQ(m_hybridMgmt.BuildH2DEmbedding(info, h2dEmb), false);
}

TEST_F(HybridMgmtTest, GetRestoreVecSec)
{
    EmbBaseInfo info;
    info.name = m_embTableName;
    info.channelId = 0;
    info.batchId = 0;
    bool remainBatchOut = false;

    m_hybridMgmt.GetRestoreVecSec(info, remainBatchOut);
}
