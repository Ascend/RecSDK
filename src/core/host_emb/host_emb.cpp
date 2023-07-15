/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Date: 2022/11/15
 */

#include "host_emb.h"
#include <utility>
#include "hd_transfer/hd_transfer.h"
#include "checkpoint/checkpoint.h"
#include "initializer/initializer.h"
#include "utils/time_cost.h"

using namespace MxRec;
using namespace std;
using namespace chrono;

bool HostEmb::Initialize(const vector<EmbInfo>& embInfos, int seed)
{
    for (const auto& embInfo: embInfos) {
        HostEmbTable hostEmb;
        hostEmb.hostEmbInfo = embInfo;
        EmbDataGenerator(embInfo.initializeInfos, seed, static_cast<int>(embInfo.hostVocabSize),
                         embInfo.extEmbeddingSize, hostEmb.embData);
        hostEmbs[embInfo.name] = move(hostEmb);
        LOG(INFO) << (HOSTEMB + "HostEmb Initialize End");
    }
    return true;
}

void HostEmb::EmbDataGenerator(const vector<InitializeInfo> &initializeInfos, int seed, int vocabSize,
    int embeddingSize, vector<vector<float>> &embData)
{
#ifndef GTEST
    LOG(INFO) << StringFormat(
        HOSTEMB + "GenerateEmbData Start, seed:%d, initializer num: %d", seed, initializeInfos.size()
    );
    embData.clear();
    embData.resize(vocabSize, vector<float>(embeddingSize));

    for (auto initializeInfo: initializeInfos) {
        Initializer* initializer;

        switch (initializeInfo.initializerType) {
            case InitializerType::CONSTANT: {
                LOG(INFO) << StringFormat(
                    HOSTEMB + "GenerateEmbData ing using Constant Initializer by value %f. name %s, "
                    "start %d, len %d.", initializeInfo.constantInitializerInfo.constantValue,
                    initializeInfo.name.c_str(), initializeInfo.start, initializeInfo.len);
                initializer = &initializeInfo.constantInitializer;
                break;
            }
            case InitializerType::TRUNCATED_NORMAL: {
                LOG(INFO) << StringFormat(
                    HOSTEMB + "GenerateEmbData ing using Truncated Normal Initializer by mean: %f stddev: %f. "
                    "name %s, start %d, len %d.", initializeInfo.normalInitializerInfo.mean,
                    initializeInfo.normalInitializerInfo.stddev, initializeInfo.name.c_str(),
                    initializeInfo.start, initializeInfo.len);
                initializer = &initializeInfo.truncatedNormalInitializer;
                break;
            }
            case InitializerType::RANDOM_NORMAL: {
                LOG(INFO) << StringFormat(
                    HOSTEMB + "GenerateEmbData ing using Random Normal Initializer by mean: %f stddev: %f. "
                    "name %s, start %d, len %d.", initializeInfo.normalInitializerInfo.mean,
                    initializeInfo.normalInitializerInfo.stddev, initializeInfo.name.c_str(),
                    initializeInfo.start, initializeInfo.len);
                initializer = &initializeInfo.randomNormalInitializer;
                break;
            }
            default: {
                LOG(WARNING) << (
                    HOSTEMB + "Invalid Initializer Type. Using default Constant Initializer with value 0.");
                ConstantInitializer defaultInitializer(initializeInfo.start, initializeInfo.len, 0, 1);
                initializer = &defaultInitializer;
            }
        }

        for (int i = 0; i < vocabSize; i++) {
            initializer->GenerateData(embData.at(i).data(), embeddingSize);
        }
    }
    LOG(INFO) << StringFormat(HOSTEMB + "GenerateEmbData End, seed:%d", seed);
#endif
}

void HostEmb::LoadEmb(emb_mem_t& loadData)
{
#ifndef GTEST
    hostEmbs = std::move(loadData);
#endif
}

void HostEmb::Join(int channelId)
{
    TimeCost tc = TimeCost();
    switch (channelId) {
        case TRAIN_CHANNEL_ID:
            VLOG(GLOG_DEBUG) << StringFormat(
                HOSTEMB + "start join, channelId:%d, procThreadsForTrain num:%d",
                channelId, procThreadsForTrain.size());
            for (auto& t: procThreadsForTrain) {
                t->join();
            }
            procThreadsForTrain.clear();
            VLOG(GLOG_DEBUG) << StringFormat(
                HOSTEMB + "end join, channelId:%d, cost:%dms", channelId, tc.ElapsedMS());
            break;
        case EVAL_CHANNEL_ID:
            VLOG(GLOG_DEBUG) << StringFormat(
                HOSTEMB + "start join, channelId:%d, procThreadsForEval num:%d",
                channelId, procThreadsForEval.size());
            for (auto& t: procThreadsForEval) {
                t->join();
            }
            procThreadsForEval.clear();
            VLOG(GLOG_DEBUG) << StringFormat(
                HOSTEMB + "end join, channelId:%d, cost:%dms", channelId, tc.ElapsedMS());
            break;
        default:
            throw invalid_argument("channelId not in [TRAIN_CHANNEL_ID, EVAL_CHANNEL_ID]");
    }
}

/*
 * 从hdTransfer获取device侧返回的emb信息，并在host侧表的对应位置插入。
 * missingKeysHostPos为host侧需要发送的emb的位置，也就是淘汰的emb的插入位置
 */
#ifndef GTEST
void HostEmb::UpdateEmb(const vector<size_t>& missingKeysHostPos, int channelId, const string& embName)
{
    LOG(INFO) << StringFormat(HOSTEMB + "UpdateEmb, channelId:%d, embName:%s", channelId, embName.c_str());
    EASY_FUNCTION(profiler::colors::Purple);
    TimeCost tc = TimeCost();
    auto hdTransfer = Singleton<MxRec::HDTransfer>::GetInstance();
    TransferChannel transferName = TransferChannel::D2H;
    LOG(INFO) << StringFormat(HOSTEMB + "wait D2H embs, channelId:%d", channelId);
    const auto tensors = hdTransfer->Recv(transferName, channelId, embName);
    if (tensors.empty()) {
        LOG(WARNING) << (HOSTEMB + "recv empty data");
        return;
    }
    const Tensor& d2hEmb = tensors[0];
    LOG(INFO) << StringFormat(HOSTEMB + "UpdateEmb End missingkeys len = %d", missingKeysHostPos.size());
    EASY_BLOCK("Update")
    const float* tensorPtr = d2hEmb.flat<float>().data();
    auto embeddingSize = hostEmbs[embName].hostEmbInfo.extEmbeddingSize;
    auto& embData = hostEmbs[embName].embData;

#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) \
                         shared(missingKeysHostPos, tensorPtr, embData, embeddingSize)
    for (size_t i = 0; i < missingKeysHostPos.size(); i++) {
        auto& dst = embData[missingKeysHostPos[i]];
#pragma omp simd
        for (int j = 0; j < embeddingSize; j++) {
            dst[j] = tensorPtr[j + embeddingSize * i];
        }
    }
    LOG(INFO) << StringFormat(HOSTEMB + "update emb end cost: %dms", tc.ElapsedMS());
    EASY_END_BLOCK
}

void HostEmb::UpdateEmbV2(const vector<size_t>& missingKeysHostPos, int channelId, const string& embName)
{
    LOG(INFO) << StringFormat(HOSTEMB + "UpdateEmbV2, channelId:%d, embName:%s", channelId, embName.c_str());
    EASY_FUNCTION(profiler::colors::Purple)
    auto updateThread =
        [&, missingKeysHostPos, channelId, embName] {
            auto hdTransfer = Singleton<MxRec::HDTransfer>::GetInstance();
            TransferChannel transferName = TransferChannel::D2H;
            LOG(INFO) << StringFormat(HOSTEMB + "wait D2H embs, channelId:%d", channelId);
            auto size = hdTransfer->RecvAcl(transferName, channelId, embName);
            if (size == 0) {
                LOG(WARNING) << (HOSTEMB + "recv empty data");
                return;
            }
            TimeCost tc = TimeCost();
            LOG(INFO) << StringFormat(HOSTEMB + "UpdateEmb End missingkeys len = %d", missingKeysHostPos.size());
            EASY_BLOCK("Update")
            auto& embData = hostEmbs[embName].embData;
            auto embeddingSize = hostEmbs[embName].hostEmbInfo.extEmbeddingSize;
            auto aclData = acltdtGetDataItem(hdTransfer->aclDatasets[embName], 0);
            if (aclData == nullptr) {
                throw runtime_error("Acl get tensor data from dataset failed.");
            }
            float* ptr = reinterpret_cast<float *>(acltdtGetDataAddrFromItem(aclData));
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) shared(ptr, embData, embeddingSize)
            for (size_t j = 0; j < missingKeysHostPos.size(); j++) {
                auto& dst = embData[missingKeysHostPos[j]];
#pragma omp simd
                for (int k = 0; k < embeddingSize; k++) {
                    dst[k] = ptr[k + embeddingSize * j];
                }
            }
            LOG(INFO) << StringFormat(HOSTEMB + "update emb end cost: %dms", tc.ElapsedMS());
    };

    switch (channelId) {
        case TRAIN_CHANNEL_ID:
            procThreadsForTrain.emplace_back(make_unique<thread>(updateThread));
            break;
        case EVAL_CHANNEL_ID:
            procThreadsForEval.emplace_back(make_unique<thread>(updateThread));
            break;
        default:
            throw invalid_argument("channelId not in [TRAIN_CHANNEL_ID, EVAL_CHANNEL_ID]");
    }
}

/*
 * 找到host侧需要发送的emb，通过hdTransfer发送给device。
 * missingKeysHostPos为host侧需要发送的emb的位置
 */
void HostEmb::GetH2DEmb(const vector<size_t>& missingKeysHostPos, const string& embName,
                        vector<Tensor>& h2dEmbOut)
{
    EASY_FUNCTION()
    TimeCost tc = TimeCost();
    const auto& emb = hostEmbs[embName];
    const int embeddingSize = emb.hostEmbInfo.extEmbeddingSize;
    h2dEmbOut.emplace_back(Tensor(tensorflow::DT_FLOAT, {
        int(missingKeysHostPos.size()), embeddingSize
    }));
    auto& tmpTensor = h2dEmbOut.back();
    auto tmpData = tmpTensor.flat<float>();
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) shared(missingKeysHostPos, emb, tmpData)
    for (size_t i = 0; i < missingKeysHostPos.size(); i++) {
        const auto& src = emb.embData[missingKeysHostPos[i]];
#pragma omp simd
        for (int j = 0; j < embeddingSize; j++) {
            tmpData(j + i * embeddingSize) = src[j];
        }
    }
    LOG(INFO) << StringFormat(
        "GetH2DEmb end, missingKeys count:%d cost:%dms", missingKeysHostPos.size(), tc.ElapsedMS());
}


auto HostEmb::GetHostEmbs() -> absl::flat_hash_map<string, HostEmbTable>*
{
    return &hostEmbs;
}

void HostEmb::EmbPartGenerator(const vector<InitializeInfo> &initializeInfos, vector<vector<float>> &embData,
                               const vector<size_t>& offset)
{
    for (auto initializeInfo: initializeInfos) {
        Initializer* initializer;

        switch (initializeInfo.initializerType) {
            case InitializerType::CONSTANT: {
                LOG(INFO) << StringFormat(HOSTEMB + "GenerateEmbData ing using Constant Initializer by value %d.",
                    initializeInfo.constantInitializerInfo.constantValue);
                initializer = &initializeInfo.constantInitializer;
                break;
            }
            case InitializerType::TRUNCATED_NORMAL: {
                LOG(INFO) << StringFormat(
                    HOSTEMB + "GenerateEmbData ing using Truncated Normal Initializer by mean: %f stddev: %f.",
                    initializeInfo.normalInitializerInfo.mean, initializeInfo.normalInitializerInfo.stddev);
                initializer = &initializeInfo.truncatedNormalInitializer;
                break;
            }
            case InitializerType::RANDOM_NORMAL: {
                LOG(INFO) << StringFormat(
                    HOSTEMB + "GenerateEmbData ing using Random Normal Initializer by mean: %f stddev: %f.",
                    initializeInfo.normalInitializerInfo.mean, initializeInfo.normalInitializerInfo.stddev);
                initializer = &initializeInfo.randomNormalInitializer;
                break;
            }
            default: {
                LOG(ERROR) << (HOSTEMB + "Invalid Initializer Type. Using default Constant Initializer with value 0.");
                ConstantInitializer defaultInitializer(initializeInfo.start, initializeInfo.len, 0, 1);
                initializer = &defaultInitializer;
            }
        }

        for (size_t i = 0; i < offset.size(); i++) {
            initializer->GenerateData(embData.at(offset.at(i)).data(), static_cast<int>(embData[0].size()));
        }
    }
}
#endif

/*
 * 利用initializer初始化emb淘汰的位置
 */
void HostEmb::EvictInitEmb(const string& embName, const vector<size_t>& offset)
{
#ifndef GTEST
    auto& hostEmb = GetEmb(embName);
    EmbPartGenerator(hostEmb.hostEmbInfo.initializeInfos, hostEmb.embData, offset);
    LOG(INFO) << StringFormat(
        HOSTEMB + "ddr EvictInitEmb!host embName %s, init offsets size: %d", embName.c_str(), offset.size());
#endif
}