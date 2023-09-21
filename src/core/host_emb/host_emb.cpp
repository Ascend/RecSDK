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

/// 初始化host emb
/// \param embInfos 表信息列表
/// \param seed 随机种子
/// \return
void HostEmb::Initialize(const vector<EmbInfo>& embInfos, int seed)
{
    for (const auto& embInfo: embInfos) {
        HostEmbTable hostEmb;
        hostEmb.hostEmbInfo = embInfo;
        EmbDataGenerator(embInfo.initializeInfos, seed, static_cast<int>(embInfo.hostVocabSize),
                         embInfo.extEmbeddingSize, hostEmb.embData);
        hostEmbs[embInfo.name] = move(hostEmb);
        LOG_INFO(HOSTEMB + "HostEmb Initialize End");
    }
}

/// 根据指定的初始化器对emb进行初始化
/// \param initializeInfos emb初始化信息列表
/// \param seed 随机种子
/// \param vocabSize host表大小
/// \param embeddingSize emb维度
/// \param embData emb数据
void HostEmb::EmbDataGenerator(const vector<InitializeInfo> &initializeInfos, int seed, int vocabSize,
    int embeddingSize, vector<vector<float>> &embData) const
{
#ifndef GTEST
    LOG_INFO(HOSTEMB + "GenerateEmbData Start, seed:{}, initializer num: {}", seed, initializeInfos.size());
    embData.clear();
    embData.resize(vocabSize, vector<float>(embeddingSize));

    for (auto initializeInfo: initializeInfos) {
        LOG_INFO("Device GenerateEmbData ing. name {}", initializeInfo.name);
        for (int i = 0; i < vocabSize; i++) {
            initializeInfo.initializer->GenerateData(embData.at(i).data(), embeddingSize);
        }
    }
    LOG_INFO(HOSTEMB + "GenerateEmbData End, seed:{}", seed);
#endif
}

/// 停止用于异步更新D2H emb的线程
/// \param channelId 通道索引（训练/推理）
void HostEmb::Join(int channelId)
{
    TimeCost tc = TimeCost();
    switch (channelId) {
        case TRAIN_CHANNEL_ID:
            LOG_DEBUG(HOSTEMB + "start join, channelId:{}, procThreadsForTrain num:{}",
                channelId, procThreadsForTrain.size());
            for (auto& t: procThreadsForTrain) {
                t->join();
            }
            procThreadsForTrain.clear();
            LOG_DEBUG(HOSTEMB + "end join, channelId:{}, cost:{}ms", channelId, tc.ElapsedMS());
            break;
        case EVAL_CHANNEL_ID:
            LOG_DEBUG(HOSTEMB + "start join, channelId:{}, procThreadsForEval num:{}",
                channelId, procThreadsForEval.size());
            for (auto& t: procThreadsForEval) {
                t->join();
            }
            procThreadsForEval.clear();
            LOG_DEBUG(HOSTEMB + "end join, channelId:{}, cost:{}ms", channelId, tc.ElapsedMS());
            break;
        default:
            throw invalid_argument("channelId not in [TRAIN_CHANNEL_ID, EVAL_CHANNEL_ID]");
    }
}

#ifndef GTEST
/// 从hdTransfer获取device侧返回的emb信息，并在host侧表的对应位置插入。
/// missingKeysHostPos为host侧需要发送的emb的位置，也就是淘汰的emb的插入位置
/// \param missingKeysHostPos 当前batch在host上需要换出的偏移
/// \param channelId 通道索引（训练/推理）
/// \param embName 表名
void HostEmb::UpdateEmb(const vector<size_t>& missingKeysHostPos, int channelId, const string& embName)
{
    LOG_INFO(HOSTEMB + "UpdateEmb, channelId:{}, embName:{}", channelId, embName);
    EASY_FUNCTION(profiler::colors::Purple);
    TimeCost tc = TimeCost();
    auto hdTransfer = Singleton<MxRec::HDTransfer>::GetInstance();
    TransferChannel transferName = TransferChannel::D2H;
    LOG_INFO(HOSTEMB + "wait D2H embs, channelId:{}", channelId);
    const auto tensors = hdTransfer->Recv(transferName, channelId, embName);
    if (tensors.empty()) {
        LOG_WARN(HOSTEMB + "recv empty data");
        return;
    }
    const Tensor& d2hEmb = tensors[0];
    EASY_BLOCK("Update")
    const float* tensorPtr = d2hEmb.flat<float>().data();
    auto embeddingSize = hostEmbs[embName].hostEmbInfo.extEmbeddingSize;
    auto& embData = hostEmbs[embName].embData;

    LOG_DEBUG(HOSTEMB + "embName:{}, UpdateEmb missingKeys len = {}, embeddingSize = {}, "
        "embData.size = {}", embName, missingKeysHostPos.size(), embeddingSize, embData.size());

#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) \
                         shared(missingKeysHostPos, tensorPtr, embData, embeddingSize)
    for (size_t i = 0; i < missingKeysHostPos.size(); i++) {
        auto& dst = embData[missingKeysHostPos[i]];
#pragma omp simd
        for (int j = 0; j < embeddingSize; j++) {
            dst[j] = tensorPtr[j + embeddingSize * i];
        }
    }
    LOG_INFO(HOSTEMB + "update emb end cost: {}ms", tc.ElapsedMS());
    EASY_END_BLOCK
}

/// 用从device获取的数据更新host的emb（使用aclTDT原生接口）
/// \param missingKeysHostPos 当前batch在host上需要换出的偏移
/// \param channelId 通道索引（训练/推理）
/// \param embName 表名
void HostEmb::UpdateEmbV2(const vector<size_t>& missingKeysHostPos, int channelId, const string& embName)
{
    LOG_INFO(HOSTEMB + "UpdateEmbV2, channelId:{}, embName:{}", channelId, embName);
    EASY_FUNCTION(profiler::colors::Purple)
    auto updateThread =
        [this, missingKeysHostPos, channelId, embName] {
            auto hdTransfer = Singleton<MxRec::HDTransfer>::GetInstance();
            TransferChannel transferName = TransferChannel::D2H;
            LOG_INFO(HOSTEMB + "wait D2H embs, channelId:{}", channelId);
            auto size = hdTransfer->RecvAcl(transferName, channelId, embName);
            if (size == 0) {
                LOG_WARN(HOSTEMB + "recv empty data");
                return;
            }
            TimeCost tc = TimeCost();

            EASY_BLOCK("Update")
            auto& embData = hostEmbs[embName].embData;
            auto embeddingSize = hostEmbs[embName].hostEmbInfo.extEmbeddingSize;
            auto aclData = acltdtGetDataItem(hdTransfer->aclDatasets[embName], 0);
            if (aclData == nullptr) {
                throw runtime_error("Acl get tensor data from dataset failed.");
            }
            float* ptr = reinterpret_cast<float *>(acltdtGetDataAddrFromItem(aclData));

            size_t elementSize = acltdtGetDataSizeFromItem(aclData);
            size_t dimNum = acltdtGetDimNumFromItem(aclData);
            LOG_DEBUG(HOSTEMB + "embName:{}, UpdateEmb missingKeys len = {}, embeddingSize = {},"
                " embData.size = {}, RecvAcl = {}, elementSize = {}, dimNum = {}",
                embName, missingKeysHostPos.size(), embeddingSize, embData.size(), size, elementSize, dimNum);
#pragma omp parallel for num_threads(MGMT_CPY_THREADS) default(none) shared(ptr, embData, embeddingSize)
            for (size_t j = 0; j < missingKeysHostPos.size(); j++) {
                auto& dst = embData[missingKeysHostPos[j]];
#pragma omp simd
                for (int k = 0; k < embeddingSize; k++) {
                    dst[k] = ptr[k + embeddingSize * j];
                }
            }
            LOG_INFO(HOSTEMB + "update emb end cost: {}ms", tc.ElapsedMS());
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

/// 查找host侧需要发送给device的emb数据。
/// \param missingKeysHostPos 当前batch在host上需要换出的偏移
/// \param embName
/// \param h2dEmbOut
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
    LOG_INFO("GetH2DEmb end, missingKeys count:{} cost:{}ms", missingKeysHostPos.size(), tc.ElapsedMS());
}

/// 获取hostEmbs的指针
/// \return
auto HostEmb::GetHostEmbs() -> absl::flat_hash_map<string, HostEmbTable>*
{
    return &hostEmbs;
}

/// 对指定offset的emb进行初始化
/// \param initializeInfos emb初始化信息列表
/// \param embData emb数据
/// \param offset 偏移列表
void HostEmb::EmbPartGenerator(const vector<InitializeInfo> &initializeInfos, vector<vector<float>> &embData,
                               const vector<size_t>& offset) const
{
    for (auto initializeInfo: initializeInfos) {
        LOG_INFO("Device GenerateEmbData ing. name {}", initializeInfo.name);
        for (size_t i = 0; i < offset.size(); i++) {
            initializeInfo.initializer->GenerateData(embData.at(offset.at(i)).data(),
                static_cast<int>(embData[0].size()));
        }
    }
}
#endif

/// 利用initializer初始化emb淘汰的位置
/// \param embName 表名
/// \param offset 淘汰的偏移列表
void HostEmb::EvictInitEmb(const string& embName, const vector<size_t>& offset)
{
#ifndef GTEST
    auto& hostEmb = GetEmb(embName);
    EmbPartGenerator(hostEmb.hostEmbInfo.initializeInfos, hostEmb.embData, offset);
    LOG_INFO(HOSTEMB + "ddr EvictInitEmb!host embName {}, init offsets size: {}", embName, offset.size());
#endif
}