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
#include <memory>

#include "common/common.h"
#include "emb_table/emb_table.h"
#include "feature_filter/feature_filter.h"
#include "swap_manager.h"
#include "utils/async_task.h"
#include "utils/thread_pool.h"
#include "file_system/file_system_handler.h"

using namespace MxRec;

namespace Embcache {

constexpr int ONE_TIME_IO_WRITE = 100000;
constexpr int SWAP_INFO_TUPLE_INDEX0 = 0;
constexpr int SWAP_INFO_TUPLE_INDEX1 = 1;
constexpr int SWAP_INFO_TUPLE_INDEX2 = 2;
constexpr int SWAP_INFO_TUPLE_INDEX3 = 3;
constexpr int SWAP_INFO_TUPLE_INDEX4 = 4;
constexpr int READ_FILE_FAILED = -1;
constexpr size_t MAX_EMB_TABLE_NUM = 10000;
constexpr size_t TABLE_NAME_LENGTH = 100;
constexpr size_t READ_AND_WRITE_SIZE_PEER_TIME = 32768;
constexpr size_t MAX_MOMENTUM_NUM = 1024;

const std::string RANK_STR_PATH = "/rank";
const std::string EMBEDDING_STR_PATH = "/embedding";
const std::string KEY_STR_PATH = "/key";
const std::string ADMIT_STR_PATH = "/admit_count";
const std::string EVICT_STR_PATH = "/evict_timestamp";
const std::string MOMENTUM1_STR_PATH = "/momentum1";
const std::string MOMENTUM2_STR_PATH = "/momentum2";
const std::string SLICE_ATTR_PATH = "/slice.attribute";
const std::string SLICE_DATA_PATH = "/slice.data";
const std::string SLICE_EVICT_KEY_DATA_PATH = "/slice_evict_key.data";
const std::string SLICE_EVICT_TS_DATA_PATH = "/slice_evict_ts.data";

constexpr int KEY_ATTRIBUTE_DATA_LEN = 2;
constexpr int EMB_ATTRIBUTE_DATA_LEN = 3;
constexpr int64_t ATTR_VEC_INIT_VALUE = -1;
constexpr long long KEY_SIZE_MAX = 1e9L;
constexpr int32_t MAX_EMB_DIM = 4096;
// load embedding count for one loop; about 5GB in size when use Adam: 100000*4096*4*3/(1024**3) = 4.58GB
constexpr int32_t ONE_TIME_LOAD_DIM_4096 = 100000;
const std::string ATTR_SUFFIX = "attribute";
const std::string DATA_SUFFIX = "data";

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

struct TableRankParam {
    TableRankParam(const std::string& tableName, int32_t tableIndex, int32_t embDim, int rank)
        : tableName(tableName), tableIndex(tableIndex), embDim(embDim), rank(rank) {}
    std::string tableName;
    int32_t tableIndex;
    int32_t embDim;
    int rank;

    // load embedding offset every for loop, only use for loading data.
    int64_t loadEmbeddingOffset = 0;
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

    void Save(const std::string& path, const int rank);

    void Embedding2Host(const at::Tensor& weightsDev, const std::vector<at::Tensor>& momentumDev);

    void Load(const std::string& path, int rank);

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

    void WriteAttributeFile(int32_t tableIndex, const std::string& pathPrefix, size_t count,
                            const std::shared_ptr<FileSystem>& fileSystemPtr);
    void CreateMomentumDir(const std::string& pathPrefix, const std::shared_ptr<FileSystem>& fileSystemPtr) const;
    void Check4Write(const std::shared_ptr<FileSystem>& fileSystemPtr, const std::string& filePath, int rank);
    void WriteData(const std::shared_ptr<FileSystem>& fileSystemPtr, const std::string& filePath, const char* dataAddr,
                   size_t dataSize);
    static std::shared_ptr<FileSystem> GetFileSystem(const std::string& path);

    template <class T>
    void ReadKeysData(const std::shared_ptr<FileSystem>& fileSystemPtr, std::vector<T>& keys,
                      const string& keyAttrFile, const string& keyDataFile);
    void ReadAttributeData(const std::shared_ptr<FileSystem>& fileSystemPtr, const string& filePath,
                           std::vector<int64_t>& dataVec, int dataCount);
    void CheckEmbeddingDim(const std::shared_ptr<FileSystem>& fileSystemPtr, const string& dataFilePath,
                           const TableRankParam& tableParams);
    void ReadEmbeddings(const std::shared_ptr<FileSystem>& fileSystemPtr, std::vector<std::vector<float>>& embeddings,
                        const string& filePath, size_t vectorSize, const TableRankParam& tableParams);
    static void RecordLoadDebugInfo(const vector<int64_t>& keys, const vector<std::vector<float>>& embeddings,
                                    const vector<std::vector<float>>& momentum1,
                                    const vector<std::vector<float>>& momentum2, const TableRankParam& tableParams);
    static std::string GetDevWeightsShape(const at::Tensor& weightsDev);

    void SaveFeatureAdmitAndEvictInfo(const std::shared_ptr<FileSystem>& fileSystemPtr,
                                      int32_t tableIndex, const std::string& filePrefix,
                                      const std::vector<int64_t>& saveKeys);
    void SaveFeatureCount(const std::shared_ptr<FileSystem>& fileSystemPtr,
                          int32_t tableIndex, const std::string& filePrefix, const std::vector<int64_t>& saveKeys);
    void SaveFeatureTimestamp(const std::shared_ptr<FileSystem>& fileSystemPtr,
                              int32_t tableIndex, const std::string& filePrefix);
    void LoadFeatureAdmitAndEvictInfo(const std::shared_ptr<FileSystem>& fileSystemPtr,
                                      int32_t tableIndex, const std::string& filePrefix,
                                      const std::vector<int64_t>& saveKeys);
    static int32_t GetOneTimeLoadCount(int32_t embDim);
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
    void LoadEmbeddingAndOptimizer(const shared_ptr<FileSystem>& fileSystemPtr, int32_t tableIndex,
                                   const string& filePrefix, const vector<int64_t>& keys,
                                   const TableRankParam& tableParams);
};
}  // namespace Embcache
#endif  // EMBEDDING_CACHE_EMBEDDING_MANAGER_H