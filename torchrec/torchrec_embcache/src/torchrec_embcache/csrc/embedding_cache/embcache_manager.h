/*
 * Copyright (c) huawei Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#ifndef EMBEDDING_CACHE_EMBEDDING_MANAGER_H
#define EMBEDDING_CACHE_EMBEDDING_MANAGER_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <torch/extension.h>
#include <vector>

#include "common/common.h"
#include "emb_table/emb_table.h"
#include "feature_filter/feature_filter.h"
#include "swap_manager.h"
#include "utils/async_task.h"
#include "utils/thread_pool.h"

namespace Embcache {

constexpr int ONE_TIME_IO_WRITE = 100000;
constexpr int SWAP_INFO_TUPLE_INDEX0 = 0;
constexpr int SWAP_INFO_TUPLE_INDEX1 = 1;
constexpr int SWAP_INFO_TUPLE_INDEX2 = 2;
constexpr int SWAP_INFO_TUPLE_INDEX3 = 3;
constexpr int SWAP_INFO_TUPLE_INDEX4 = 4;
constexpr size_t TABLE_NAME_LENGTH = 100;
constexpr size_t READ_AND_WRITE_SIZE_PEER_TIME = 32768;


struct SwapInfo {
    std::vector<std::vector<int64_t>> swapoutKeys;
    at::Tensor swapoutOffs;
    std::vector<std::vector<int64_t>> swapinKeys;
    at::Tensor swapinOffs;
    at::Tensor batchOffs;
    std::vector<int64_t> swapinKeysLengthPreSum;
    std::vector<int64_t> swapoutKeysLengthPreSum;

    const std::vector<int64_t>& GetSwapinKeysLengthPreSum()
    {
        if (swapinKeysLengthPreSum.empty()) {
            uint64_t preSum = 0;
            swapinKeysLengthPreSum.emplace_back(preSum);
            for (const auto& keys : swapinKeys) {
                preSum += keys.size();
                swapinKeysLengthPreSum.emplace_back(preSum);
            }
        }
        return swapinKeysLengthPreSum;
    }

    const std::vector<int64_t>& GetSwapoutKeysLengthPreSum()
    {
        if (swapoutKeysLengthPreSum.empty()) {
            uint64_t preSum = 0;
            swapoutKeysLengthPreSum.reserve(swapoutKeys.size() + 1);
            swapoutKeysLengthPreSum.emplace_back(preSum);
            for (const auto& keys : swapoutKeys) {
                preSum += keys.size();
                swapoutKeysLengthPreSum.emplace_back(preSum);
            }
        }
        return swapoutKeysLengthPreSum;
    }
};

struct SwapinTensor {
    at::Tensor swapinEmbs;
    std::vector<at::Tensor> swapinOptims;
    at::Tensor jaggedOffs;                 // 区分每个表
};

class EmbcacheManager {
public:
    explicit EmbcacheManager(const std::vector<EmbConfig>& embConfigs, bool needAccumulateOffset = true);

    ~EmbcacheManager()
    {
        GetEmbMemoryPoolThreadPool().Stop();
        GetAsyncTaskPool().Stop();
    }

    EmbcacheManager(const EmbcacheManager& cacheManager) = delete;

    EmbcacheManager& operator=(const EmbcacheManager& cacheManager) = delete;

    AsyncTask<SwapInfo> ComputeSwapInfoAsync(const at::Tensor& batchKeys, const std::vector<int64_t>& offsetPerKey,
                                             const std::vector<int32_t>& tableIndices);

    AsyncTask<SwapinTensor> EmbeddingLookupAsync(const SwapInfo& swapInfo, const std::vector<int32_t>& tableIndices);

    AsyncTask<void> EmbeddingUpdateAsync(const SwapInfo& swapInfo, const at::Tensor& swapoutEmbs,
                                         const std::vector<at::Tensor>& swapoutOptims,
                                         const std::vector<int32_t>& tableIndices);

    void EvictFeatures();

    void RecordTimestamp(const at::Tensor& batchKeys, const std::vector<int64_t>& offsetPerKey,
                         const at::Tensor& timestamps, const std::vector<int32_t>& tableIndices);

    void StatisticsKeyCount(const at::Tensor& batchKeys, const torch::Tensor& offset, const at::Tensor& batchKeyCounts,
                            int64_t tableIndex);

    void RecordEmbeddingUpdateTimes();

private:
    SwapInfo ComputeSwapInfo(const at::Tensor& batchKeys, const std::vector<int64_t>& offsetPerKey,
                             const std::vector<int32_t>& tableIndices);

    SwapinTensor EmbeddingLookup(const std::vector<std::vector<int64_t>>& swapinKeys,
                                 const std::vector<int32_t>& tableIndices);

    void EmbeddingUpdate(const std::vector<std::vector<int64_t>>& swapoutKeys, const at::Tensor& swapoutEmbs,
                         const std::vector<at::Tensor>& swapoutOptims, const std::vector<int32_t>& tableIndices);

    bool EnableFastHashMap();

    bool NeedEvictEmbeddingTable();
    void RemoveEmbeddingTableInfo();

private:
    int32_t embNum_;
    std::vector<int32_t> embTableIndies_;
    std::vector<EmbConfig> embConfigs_;
    std::vector<SwapManager> swapManagers_;
    std::vector<std::unique_ptr<EmbTable>> embeddingTables_;
    std::vector<std::unique_ptr<FeatureFilter>> featureFilters_;  // 索引直接对应表索引，未启用的为nullptr

    uint64_t swapCount_ = 0;       // ComputeSwapInfo 执行次数
    uint64_t embUpdateCount_ = 0;  // EmbeddingUpdate 执行次数

    bool enableFastHashMap_ = false;
    int32_t optimNum_;

    // 计算换入换出offset时是否要累加表外偏移. 逻辑上作为一个大表处理时设置为true，否则false
    bool needAccumulateOffset_ = true;
};
}  // namespace Embcache
#endif  // EMBEDDING_CACHE_EMBEDDING_MANAGER_H