/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Date: 2022/11/15
 */

#include "host_emb.h"
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <utility>
#include "hd_transfer/hd_transfer.h"
#include "checkpoint/checkpoint.h"
#include "initializer/initializer.h"

using namespace MxRec;
using namespace std;
using namespace chrono;

bool HostEmb::Initialize(const vector<EmbInfo>& embInfos, int seed, bool ifLoad)
{
    for (const auto& embInfo: embInfos) {
        HostEmbTable hostEmb;
        hostEmb.hostEmbInfo = embInfo;
        EmbDataGenerator(embInfo.initializeInfos, seed, embInfo.hostVocabSize, embInfo.extEmbeddingSize,
            hostEmb.embData);
        hostEmbs[embInfo.name] = move(hostEmb);
        spdlog::info(HOSTEMB + "HostEmb Initialize End");
    }
    return true;
}

void HostEmb::EmbDataGenerator(const vector<InitializeInfo> &initializeInfos, int seed, int vocabSize,
    int embeddingSize, vector<vector<float>> &embData)
{
    spdlog::info(HOSTEMB + "GenerateEmbData Start, seed:{}", seed);
    embData.clear();
    embData.resize(vocabSize, vector<float>(embeddingSize));

    for (auto initializeInfo: initializeInfos) {
        Initializer* initializer;

        switch (initializeInfo.initializerType) {
            case InitializerType::CONSTANT: {
                spdlog::info(HOSTEMB + "GenerateEmbData ing using Constant Initializer by value {}.",
                    initializeInfo.constantInitializerInfo.constantValue);
                initializer = &initializeInfo.constantInitializer;
                break;
            }
            case InitializerType::TRUNCATED_NORMAL: {
                spdlog::info(HOSTEMB + "GenerateEmbData ing using Truncated Normal Initializer by mean: {} stddev: {}.",
                    initializeInfo.normalInitializerInfo.mean, initializeInfo.normalInitializerInfo.stddev);
                initializer = &initializeInfo.truncatedNormalInitializer;
                break;
            }
            case InitializerType::RANDOM_NORMAL: {
                spdlog::info(HOSTEMB + "GenerateEmbData ing using Random Normal Initializer by mean: {} stddev: {}.",
                    initializeInfo.normalInitializerInfo.mean, initializeInfo.normalInitializerInfo.stddev);
                initializer = &initializeInfo.randomNormalInitializer;
                break;
            }
            default: {
                spdlog::warn(HOSTEMB + "Invalid Initializer Type. Using default Constant Initializer with value 0.");
                ConstantInitializer defaultInitializer(initializeInfo.start, initializeInfo.len, 0);
                initializer = &defaultInitializer;
            }
        }

        for (int i = 0; i < vocabSize; i++) {
            initializer->GenerateData(embData.at(i).data(), embeddingSize);
        }
    }

    spdlog::info(HOSTEMB + "GenerateEmbData End, seed:{}", seed);
}

void HostEmb::LoadEmb(emb_mem_t& loadData)
{
    hostEmbs = std::move(loadData);
}

void HostEmb::Join()
{
    spdlog::stopwatch sw;
    spdlog::debug(HOSTEMB + "hostemb start join {}", procThreads.size());
    for (auto& t: procThreads) {
        t->join();
    }
    procThreads.clear();
    spdlog::info(HOSTEMB + "hostemb end join, cost:{}", TO_MS(sw));
}

/*
 * 从hdTransfer获取device侧返回的emb信息，并在host侧表的对应位置插入。
 * missingKeysHostPos为host侧需要发送的emb的位置，也就是淘汰的emb的插入位置
 */
void HostEmb::UpdateEmb(const vector<size_t>& missingKeysHostPos, int channelId, const string& embName)
{
    EASY_FUNCTION(profiler::colors::Purple)
    auto hdTransfer = Singleton<MxRec::HDTransfer>::GetInstance();
    TransferChannel transferName = TransferChannel::D2H;
    spdlog::info(HOSTEMB + "wait D2H embs, channelId:{}", channelId);
    const auto tensors = hdTransfer->Recv(transferName, channelId, embName);
    if (tensors.empty()) {
        spdlog::warn(HOSTEMB + "recv empty data");
        return;
    }
    const Tensor& d2hEmb = tensors[0];
    spdlog::info(HOSTEMB + "UpdateEmb End missingkeys len = {}", missingKeysHostPos.size());
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
            dst[j] = tensorPtr[j];
        }
        tensorPtr = tensorPtr + embeddingSize;
    }
    spdlog::info(HOSTEMB + "update emb end");
    EASY_END_BLOCK
}

void HostEmb::UpdateEmbV2(const vector<size_t>& missingKeysHostPos, int channelId, const string& embName)
{
#ifndef GTEST
    EASY_FUNCTION(profiler::colors::Purple)
    procThreads.emplace_back(make_unique<thread>(
        [&, missingKeysHostPos, channelId, embName] {
            auto hdTransfer = Singleton<MxRec::HDTransfer>::GetInstance();
            TransferChannel transferName = TransferChannel::D2H;
            spdlog::info(HOSTEMB + "wait D2H embs, channelId:{}", channelId);
            auto [aclDataset, size] = hdTransfer->RecvAcl(transferName, channelId, embName);
            if (size == 0) {
                spdlog::warn(HOSTEMB + "recv empty data");
                return;
            }
            spdlog::info(HOSTEMB + "UpdateEmb End missingkeys len = {}", missingKeysHostPos.size());
            EASY_BLOCK("Update")
            auto& embData = hostEmbs[embName].embData;
            auto embeddingSize = hostEmbs[embName].hostEmbInfo.extEmbeddingSize;
            auto aclData = acltdtGetDataItem(aclDataset, 0);
            if (aclData == nullptr) {
                throw runtime_error("Acl get tensor data from dataset failed.");
            }
            float* ptr = reinterpret_cast<float *>(acltdtGetDataAddrFromItem(aclData));
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) shared(ptr, embData, embeddingSize)
            for (size_t j = 0; j < missingKeysHostPos.size(); j++) {
                auto& dst = embData[missingKeysHostPos[j]];
#pragma omp simd
                for (int k = 0; k < embeddingSize; k++) {
                    dst[k] = ptr[k];
                }
            }
            if (acltdtDestroyDataset(aclDataset) != ACL_ERROR_NONE) {
                throw runtime_error("Acl destroy tensor dataset failed.");
            }
            spdlog::info(HOSTEMB + "update emb end");
        }));
#endif
}

/*
 * 找到host侧需要发送的emb，通过hdTransfer发送给device。
 * missingKeysHostPos为host侧需要发送的emb的位置
 */
void HostEmb::GetH2DEmb(const vector<size_t>& missingKeysHostPos, const string& embName,
                        vector<Tensor>& h2dEmbOut)
{
    EASY_FUNCTION()
    const auto& emb = hostEmbs[embName];
    const int embeddingSize = emb.hostEmbInfo.extEmbeddingSize;
    h2dEmbOut.emplace_back(Tensor(tensorflow::DT_FLOAT, {
        int(missingKeysHostPos.size()), embeddingSize
    }));
    auto& tmpTensor = h2dEmbOut.back();
    auto tmpData = tmpTensor.flat<float>();
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) shared(missingKeysHostPos, emb, tmpData)
    for (size_t i = 0; i < missingKeysHostPos.size(); i++) {
        const auto src = emb.embData[missingKeysHostPos[i]];
#pragma omp simd
        for (int j = 0; j < embeddingSize; j++) {
            tmpData(j + i * embeddingSize) = src[j];
        }
    }
    spdlog::info("GetH2DEmb end, missingKeys count:{}", missingKeysHostPos.size());
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
                spdlog::info(HOSTEMB + "GenerateEmbData ing using Constant Initializer by value {}.",
                             initializeInfo.constantInitializerInfo.constantValue);
                initializer = &initializeInfo.constantInitializer;
                break;
            }
            case InitializerType::TRUNCATED_NORMAL: {
                spdlog::info(HOSTEMB + "GenerateEmbData ing using Truncated Normal Initializer by mean: {} stddev: {}.",
                             initializeInfo.normalInitializerInfo.mean, initializeInfo.normalInitializerInfo.stddev);
                initializer = &initializeInfo.truncatedNormalInitializer;
                break;
            }
            case InitializerType::RANDOM_NORMAL: {
                spdlog::info(HOSTEMB + "GenerateEmbData ing using Random Normal Initializer by mean: {} stddev: {}.",
                             initializeInfo.normalInitializerInfo.mean, initializeInfo.normalInitializerInfo.stddev);
                initializer = &initializeInfo.randomNormalInitializer;
                break;
            }
            default: {
                spdlog::error(HOSTEMB + "Invalid Initializer Type. Using default Constant Initializer with value 0.");
                ConstantInitializer defaultInitializer(initializeInfo.start, initializeInfo.len, 0);
                initializer = &defaultInitializer;
            }
        }

        for (size_t i = 0; i < offset.size(); i++) {
            initializer->GenerateData(embData.at(offset.at(i)).data(), embData[0].size());
        }
    }
}

/*
 * 利用initializer初始化emb淘汰的位置
 */
void HostEmb::EvictInitEmb(const string& embName, const vector<size_t>& offset)
{
    auto& hostEmb = GetEmb(embName);
    EmbPartGenerator(hostEmb.hostEmbInfo.initializeInfos, hostEmb.embData, offset);

    spdlog::info(HOSTEMB + "ddr EvictInitEmb!host embName {}, init offsets size: {}", embName, offset.size());
}
