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
#include <string>
#include <torch/extension.h>
#include <vector>

#include "common/common.h"
#include "emb_table/emb_table.h"
#include "feature_filter/feature_filter.h"
#include "swap_manager.h"
#include "utils/async_task.h"

namespace Embcache {

constexpr int ONE_TIME_IO_WRITE = 100000;
constexpr int SWAP_INFO_TUPLE_INDEX0 = 0;
constexpr int SWAP_INFO_TUPLE_INDEX1 = 1;
constexpr int SWAP_INFO_TUPLE_INDEX2 = 2;
constexpr int SWAP_INFO_TUPLE_INDEX3 = 3;
constexpr int SWAP_INFO_TUPLE_INDEX4 = 4;
constexpr size_t TABLE_NAME_LENGTH = 100;
constexpr size_t READ_AND_WRITE_SIZE_PEER_TIME = 32768;

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

struct SwapInfo {
    std::vector<std::vector<int64_t>> swapoutKeys;
    at::Tensor swapoutOffs;
    std::vector<std::vector<int64_t>> swapinKeys;
    at::Tensor swapinOffs;
    at::Tensor batchOffs;
    std::vector<int64_t> swapinKeysLength;
    std::vector<int64_t> swapoutKeysLength;
    const std::vector<int64_t>& getSwapinKeysLength()
    {
        if (swapinKeysLength.empty()) {
            for (const auto& keys : swapinKeys) {
                swapinKeysLength.emplace_back(keys.size());
            }
        }
        return swapinKeysLength;
    }
    const std::vector<int64_t>& getSwapoutKeysLength()
    {
        if (swapoutKeysLength.empty()) {
            for (const auto& keys : swapoutKeys) {
                swapoutKeysLength.emplace_back(keys.size());
            }
        }
        return swapoutKeysLength;
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

    AsyncTask<SwapInfo> ComputeSwapInfoAsync(const at::Tensor& batchKeys, const std::vector<int64_t>& offsetPerKey);

    AsyncTask<SwapinTensor> EmbeddingLookupAsync(const SwapInfo& swapInfo);

    AsyncTask<void> EmbeddingUpdateAsync(const SwapInfo& swapInfo, const at::Tensor& swapoutEmbs,
                                         const std::vector<at::Tensor>& swapoutOptims);

    void Save(const std::string path, const int rank);

    void Embedding2Host(const at::Tensor& weightsDev, const std::vector<at::Tensor>& momentumDev);

    void Load(const std::string& path, int rank);

    void EvictFeatures();

    void RecordTimestamp(const at::Tensor& batchKeys, const std::vector<int64_t>& offsetPerKey,
                         const at::Tensor& timestamps, const std::vector<int32_t>& tableIndices = {});

    void StatisticsKeyCount(const at::Tensor& batchKeys, const torch::Tensor& offset, const at::Tensor& batchKeyCounts,
                            int64_t tableIndex);

    void RecordEmbeddingUpdateTimes();

    /**
     * 读取指定文件。 示例：save_dir/sparse/table1/rank0/key/slice.data
     * @tparam T 数据类型泛型
     * @param filePath 示例：save_dir/sparse/table1/rank0
     * @param dataOutputs 输出参数，读取到的数据集合
     * @param loadItemName 读取哪一种类别文件，示例：key
     * @param detailFileName 具体文件名称，示例：/slice.data
     * @return code
     */
    template <class T>
    static int32_t ReadFile(const std::string& filePath, std::vector<T>& dataOutputs, const std::string& loadItemName,
                            const std::string& detailFileName = "/slice.data");

    static int32_t ReadFile(const std::string& filePath, std::vector<std::vector<float>>& embedding,
                            const std::string& loadItemName, int32_t embDim);

private:
    SwapInfo ComputeSwapInfo(const at::Tensor& batchKeys, const std::vector<int64_t>& offsetPerKey, 
                             const std::vector<int32_t>& tableIndices = {});

    SwapinTensor EmbeddingLookup(const std::vector<std::vector<int64_t>>& swapinKeys,
                                 const std::vector<int32_t>& tableIndices = {});

    void EmbeddingUpdate(const std::vector<std::vector<int64_t>>& swapoutKeys, const at::Tensor& swapoutEmbs,
                         const std::vector<at::Tensor>& swapoutOptims, const std::vector<int32_t>& tableIndices = {});

    bool EnableFastHashMap();
    std::ofstream OpenFile(std::string path);
    void WriteData(std::ofstream& file, const char* dataPtr, size_t bytes);

    bool NeedEvictEmbeddingTable();
    void RemoveEmbeddingTableInfo();
    void SaveFeatureAdmitAndEvictInfo(int32_t tableIndex, const std::string& filePrefix,
                                      const std::vector<int64_t>& saveKeys);
    void LoadFeatureAdmitAndEvictInfo(int32_t tableIndex, const std::string& filePrefix,
                                      const std::vector<int64_t>& saveKeys);
    std::string GetDevWeightsShape(const at::Tensor& weightsDev) const;
    void WriteOptimizerAttributeFile(int32_t i, std::ofstream& fileMomentum1SliceAttr,
                                     std::ofstream& fileMomentum2SliceAttr, size_t count);
    void SaveFeatureCount(int32_t tableIndex, const std::string& filePrefix, const std::vector<int64_t>& saveKeys);
    void SaveFeatureTimestamp(int32_t tableIndex, const std::string& filePrefix);
    int32_t GetEmbTableIndex(const std::string& tableName);

private:
    int32_t embNum;
    std::map<int32_t, int32_t> embTableIndexMap_;
    std::vector<EmbConfig> embConfigs;
    std::vector<SwapManager> swapManagers;
    std::vector<std::unique_ptr<EmbTable>> embeddingTables;
    std::vector<FeatureFilter> featureFilters;

    uint64_t swapCount = 0;       // ComputeSwapInfo 执行次数
    uint64_t embUpdateCount = 0;  // EmbeddingUpdate 执行次数

    bool enableFastHashMap = false;
    int32_t optimNum;

    // 计算换入换出offset时是否要累加表外偏移. 逻辑上作为一个大表处理时设置为true，否则false
    bool needAccumulateOffset = true;
};
}  // namespace Embcache
#endif  // EMBEDDING_CACHE_EMBEDDING_MANAGER_H