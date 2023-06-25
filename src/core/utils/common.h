/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Create: 2021
 * History: NA
 */

#ifndef COMMON_H
#define COMMON_H

#include <sys/stat.h>

#include <cstring>
#include <cassert>

#include <vector>
#include <random>
#include <chrono>
#include <mutex>
#include <map>
#include <queue>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>
#include <spdlog/fmt/chrono.h>
#include <spdlog/fmt/bundled/ranges.h>
#include <spdlog/cfg/env.h>
#include "tensorflow/core/framework/tensor.h"
#include "absl/container/flat_hash_map.h"

#include "initializer/initializer.h"
#include "initializer/constant_initializer/constant_initializer.h"
#include "initializer/truncated_normal_initializer/truncated_normal_initializer.h"
#include "initializer/random_normal_initializer/random_normal_initializer.h"

#if defined(BUILD_WITH_EASY_PROFILER)
    #include <easy/profiler.h>
    #include <easy/arbitrary_value.h>
#else
    #define EASY_FUNCTION(...)
    #define EASY_VALUE(...)
    #define EASY_BLOCK(...)
    #define EASY_END_BLOCK
    #define EASY_PROFILER_ENABLE
    #define EASY_PROFILER_DISABLE
#endif

namespace MxRec {
#define INFO_PTR shared_ptr
#define TIME_PRINT spdlog::debug
#define MGMT_CPY_THREADS 4
#define PROFILING
    using namespace tensorflow;
    constexpr int TRAIN_CHANNEL_ID = 0;
    constexpr int EVAL_CHANNEL_ID = 1;

    constexpr int MAX_CHANNEL_NUM = 2;
    constexpr int MAX_KEY_PROCESS_THREAD = 10;
    constexpr int MAX_QUEUE_NUM = MAX_CHANNEL_NUM * MAX_KEY_PROCESS_THREAD;
    constexpr int DEFAULT_KEY_PROCESS_THREAD = 6;
    constexpr int KEY_PROCESS_THREAD = 6;

    // unique related config
    constexpr int UNIQUE_BUCKET = 6;
    constexpr int MIN_UNIQUE_THREAD_NUM = 1;
    constexpr int DEFAULT_MAX_UNIQUE_THREAD_NUM = 8;

    struct PerfConfig {
        static int keyProcessThreadNum;
        static int maxUniqueThreadNum;
        static bool fastUnique;
    };

    constexpr int KEY_PROCESS_TIMEOUT = 120;
    constexpr int GET_BATCH_TIMEOUT = 300;

    constexpr size_t DEFAULT_RANDOM_SEED = 10086;
    constexpr int INVALID_KEY_VALUE = -1;
    constexpr int PROFILING_START_BATCH_ID = 100;
    constexpr int PROFILING_END_BATCH_ID = 200;
    constexpr int MGMT_THREAD_BIND = 48;
    constexpr int UNIQUE_MAX_BUCKET_WIDTH = 6;
    constexpr int HOT_EMB_UPDATE_STEP_DEFAULT = 1000;
    constexpr float HOT_EMB_CACHE_PCT = static_cast<float>(1. / 3);  // hot emb cache percent

    using emb_key_t = int64_t;
    using emb_name_t = std::string;
    using keys_t = std::vector<emb_key_t>;
    using lookup_key_t = std::tuple<int, emb_name_t, keys_t>;             // batch_id quarry_lable keys_vector
    using tensor_info_t = std::tuple<int, emb_name_t, std::list<std::unique_ptr<std::vector<Tensor>>>::iterator>;
    using EndRunError = std::runtime_error;

    namespace HybridOption {
        const int USE_STATIC = 0x001;
        const int USE_HOT = 0x001 << 1;
        const int USE_DYNAMIC_EXPANSION = 0x001 << 2;
    };

    string GetChipName(int devID);

    namespace UBSize {
        const int ASCEND910_PREMIUM_A = 262144;
        const int ASCEND910_PRO_B = 262144;
        const int ASCEND910_B2 = 196608;
        const int ASCEND910_B1 = 196608;
        const int ASCEND910_B3 = 196608;
        const int ASCEND910_B4 = 196608;
        const int ASCEND910_C1 = 196608;
        const int ASCEND910_C2 = 196608;
        const int ASCEND910_C3 = 196608;
        const int ASCEND920_A = 196608;
        const int ASCEND910_PRO_A = 262144;
        const int ASCEND910_B = 262144;
        const int ASCEND910_A = 262144;
    };

    inline int GetUBSize(int devID)
    {
        std::map<string, int> ChipUbSizeList = {{"910A", UBSize::ASCEND910_A},
                                                {"910B", UBSize::ASCEND910_B},
                                                {"920A", UBSize::ASCEND920_A},
                                                {"910B1", UBSize::ASCEND910_B1},
                                                {"910B2", UBSize::ASCEND910_B2},
                                                {"910B3", UBSize::ASCEND910_B3},
                                                {"910B4", UBSize::ASCEND910_B4}};
        auto it = ChipUbSizeList.find(GetChipName(devID));
        if (it != ChipUbSizeList.end()) {
            return it->second;
        }

        throw std::runtime_error("unknown chip ub size" + GetChipName(devID));
    }

    inline  std::chrono::milliseconds::rep Format2Ms(spdlog::stopwatch& sw)
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>((sw).elapsed()).count();
    }

    template <class T>
    struct Batch {
        size_t Size() const
        {
            return sample.size();
        }

        std::string UnParse() const
        {
            std::string s;
            constexpr size_t MAX_DISP_LEN = 20;
            int maxLen = static_cast<int>(std::min(sample.size(), MAX_DISP_LEN));
            for (int i = 0; i < maxLen; i++) {
                s += std::to_string(sample[i]) + " ";
            }
            return s;
        }

        std::vector<T> sample;
        void *tensorAddr = nullptr;
        std::string name;
        size_t batchSize;
        int batchId;
        int channel = 0;
        bool isInt64; // true int64 false int32
        time_t timestamp { -1 };
    };

struct BatchTask {
    vector<int> splits;
    vector<string> embNames;
    size_t batchSize;
    int batchQueueId;
    int batchId;
    int channelId;
    time_t timestamp { -1 };
    bool flag; // true int64 false int32
    const void *tensor;
};

    using emb_batch_t = Batch<int64_t>;
    using batch_task_t = BatchTask;


    struct RankInfo {
        RankInfo() = default;

        RankInfo(int rankId, int deviceId, int localRankSize, int option, int nBatch,
                 const std::vector<int>& maxStep);
        RankInfo(int localRankSize, int option, int nBatch, const std::vector<int>& maxStep);

        int rankId {};
        int deviceId {};
        int rankSize {};
        int localRankId {};
        int localRankSize {};
        bool useStatic { false };
        bool useHot {};
        uint32_t option {};
        int nBatch {};
        bool useDataset { false }; // deprecated
        bool noDDR { false };
        bool useDynamicExpansion {false};
        std::vector<int> maxStep;
    };

    enum TensorIndex : uint32_t {
        TENSOR_INDEX_0,
        TENSOR_INDEX_1,
        TENSOR_INDEX_2,
        TENSOR_INDEX_3,
        TENSOR_INDEX_4,
        TENSOR_INDEX_5,
        TENSOR_INDEX_6,
        TENSOR_INDEX_7,
        TENSOR_INDEX_8
    };

    enum TupleIndex : uint32_t {
        TUPLE_INDEX_0 = 0,
        TUPLE_INDEX_1,
        TUPLE_INDEX_2,
        TUPLE_INDEX_3,
        TUPLE_INDEX_4,
        TUPLE_INDEX_5,
        TUPLE_INDEX_6,
        TUPLE_INDEX_7
    };

    struct RandomInfo {
        RandomInfo() = default;

        RandomInfo(int start, int len, float constantVal, float randomMin, float randomMax);

        int start;
        int len;
        float constantVal;
        float randomMin;
        float randomMax;
    };

    struct ThresholdValue {
        ThresholdValue() = default;
        ThresholdValue(emb_name_t name, int countThre, int timeThre)
        {
            tensorName = name;
            countThreshold = countThre;
            timeThreshold = timeThre;
        }

        emb_name_t tensorName { "" }; // embName
        int countThreshold { -1 }; // 只配置count，即“只有准入、而没有淘汰”功能，对应SingleHostEmbTableStatus::SETS_ONLY_ADMIT状态
        int timeThreshold { -1 };  // 只配置time，配置错误；即准入是淘汰的前提，对应SingleHostEmbTableStatus::SETS_BOTH状态
    };

    struct FeatureItemInfo {
        FeatureItemInfo() = default;
        FeatureItemInfo(int64_t id, uint32_t cnt, std::string name, time_t lastT)
            : featureId(id), count(cnt), tensorName(name), lastTime(lastT)
        {}

        bool operator > (const FeatureItemInfo& item) const
        {
            return lastTime > item.lastTime;
        }

        int64_t featureId { -1 };
        uint32_t count { 0 };
        std::string tensorName { "" };
        time_t lastTime { 0 };
    };

    using SortedRecords =
        std::priority_queue<FeatureItemInfo, std::vector<FeatureItemInfo>, std::greater<FeatureItemInfo>>;
    using HistoryRecords = absl::flat_hash_map<std::string, absl::flat_hash_map<int64_t, FeatureItemInfo>>;
    struct AdmitAndEvictData {
        HistoryRecords historyRecords;                       // embName ---> {id, FeatureItemInfo} 映射
        absl::flat_hash_map<std::string, time_t> timestamps; // 用于特征准入&淘汰的时间戳
    };

    void SetLog(int rank);

    inline void GenerateRandomValue(std::vector<float>& vecData,
                                    std::default_random_engine& generator,
                                    RandomInfo& randomInfo)
    {
        float min = ((!randomInfo.randomMin) ? -0.1f : randomInfo.randomMin);
        float max = ((!randomInfo.randomMax) ? 0.1f : randomInfo.randomMax);
        if (randomInfo.len == 0) {
            return;
        }
        assert(static_cast<int>(vecData.size()) >= randomInfo.len + randomInfo.start);
        std::uniform_real_distribution<float> distribution(min, max);
        std::generate_n(vecData.begin() + randomInfo.start, randomInfo.len, [&]() { return distribution(generator); });
    }

    enum class InitializerType {
        CONSTANT,
        TRUNCATED_NORMAL,
        RANDOM_NORMAL
    };

    struct ConstantInitializerInfo {
        ConstantInitializerInfo() = default;
        explicit ConstantInitializerInfo(float constantValue, float initK);

        float constantValue;
        float initK = 1.0;
    };

    struct NormalInitializerInfo {
        NormalInitializerInfo() = default;
        NormalInitializerInfo(float mean, float stddev, int seed, float initK);

        float mean;
        float stddev;
        int seed;
        float initK = 1.0;
    };

    struct InitializeInfo {
        InitializeInfo() = default;

        InitializeInfo(std::string& name, int start, int len, ConstantInitializerInfo constantInitializerInfo);
        InitializeInfo(std::string& name, int start, int len, NormalInitializerInfo normalInitializerInfo);

        std::string name;
        int start;
        int len;
        InitializerType initializerType;

        ConstantInitializerInfo constantInitializerInfo;
        NormalInitializerInfo normalInitializerInfo;

        ConstantInitializer constantInitializer;
        TruncatedNormalInitializer truncatedNormalInitializer;
        RandomNormalInitializer randomNormalInitializer;
    };

    template<class T>
    inline Tensor Vec2TensorI32(const std::vector<T>& data)
    {
        Tensor tmpTensor(tensorflow::DT_INT32, { static_cast<int>(data.size()) });
        auto tmpData = tmpTensor.flat<int32>();
        for (int j = 0; j < static_cast<int>(data.size()); j++) {
            tmpData(j) = static_cast<int>(data[j]);
        }
        return tmpTensor;
    }

    template<class T>
    inline Tensor Vec2TensorI64(const std::vector<T>& data)
    {
        Tensor tmpTensor(tensorflow::DT_INT64, { static_cast<int>(data.size()) });
        auto tmpData = tmpTensor.flat<int64>();
        for (int j = 0; j < static_cast<int>(data.size()); j++) {
            tmpData(j) = static_cast<int64>(data[j]);
        }
        return tmpTensor;
    }

    struct EmbInfo {
        EmbInfo() = default;

        EmbInfo(const std::string& name,
                int sendCount,
                int embeddingSize,
                int extEmbeddingSize,
                bool isSave,
                std::vector<size_t> vocabsize,
                std::vector<InitializeInfo> initializeInfos)
            : name(name), sendCount(sendCount), embeddingSize(embeddingSize), extEmbeddingSize(extEmbeddingSize),
              isSave(isSave), initializeInfos(initializeInfos)
        {
            devVocabSize = vocabsize[0];
            hostVocabSize = vocabsize[1];
        }

        std::string name;
        int sendCount;
        int embeddingSize;
        int extEmbeddingSize;
        bool isSave;
        size_t devVocabSize;
        size_t hostVocabSize;
        std::vector<InitializeInfo> initializeInfos;
    };

    struct HostEmbTable {
        EmbInfo hostEmbInfo;
        std::vector<std::vector<float>> embData;
    };

    struct EmbHashMapInfo {
        absl::flat_hash_map<emb_key_t, size_t> hostHashMap;
        std::vector<int> devOffset2Batch; // has -1
        std::vector<emb_key_t> devOffset2Key;
        size_t currentUpdatePos;
        size_t currentUpdatePosStart;
        size_t hostVocabSize;
        size_t devVocabSize;
        size_t freeSize;
        std::vector<int32_t> lookUpVec;
        std::vector<size_t> missingKeysHostPos;
        std::vector<size_t> swapPos;
        size_t maxOffset { 0 };
        std::vector<size_t> evictPos;
        std::vector<size_t> evictDevPos;
        size_t maxOffsetOld { 0 };
        std::vector<size_t> evictPosChange;
        std::vector<size_t> evictDevPosChange;
        std::vector<std::pair<int, emb_key_t>> devOffset2KeyOld;
        std::vector<std::pair<emb_key_t, emb_key_t>> oldSwap; // (old on dev, old on host)

        void SetStartCount();

        bool HasFree(size_t i);
    };

    struct All2AllInfo {
        keys_t keyRecv;
        vector<int> scAll;
        vector<uint32_t> countRecv;
    };

    struct UniqueInfo {
        vector<int32_t> restore;
        vector<int32_t> hotPos;
        All2AllInfo all2AllInfo;
    };

    struct KeySendInfo {
        keys_t keySend;
        vector<int32_t> keyCount;
    };

    using emb_mem_t = absl::flat_hash_map<std::string, HostEmbTable>;
    using emb_hash_mem_t = absl::flat_hash_map<std::string, EmbHashMapInfo>;
    using offset_mem_t = std::map<emb_name_t, size_t>;
    using key_offset_mem_t = std::map<emb_name_t, absl::flat_hash_map<emb_key_t, int64_t>>;
    using tensor_2_thresh_mem_t = absl::flat_hash_map<std::string, ThresholdValue>;
    using trans_serialize_t = uint8_t;

    enum class CkptFeatureType {
        HOST_EMB = 0,
        EMB_HASHMAP = 1,
        MAX_OFFSET = 2,
        KEY_OFFSET_MAP = 3,
        FEAT_ADMIT_N_EVICT = 4
    };

    struct CkptData {
        emb_mem_t* hostEmbs = nullptr;
        emb_hash_mem_t embHashMaps;
        offset_mem_t maxOffset;
        key_offset_mem_t keyOffsetMap;
        tensor_2_thresh_mem_t tens2Thresh;
        AdmitAndEvictData histRec;
    };

    struct CkptTransData {
        std::vector<int64_t> int64Arr;
        std::vector<float*> floatArr;
        std::vector<int32_t> int32Arr;
        std::vector<trans_serialize_t> transDataset; // may all use this to transfer data
        std::vector<size_t> attribute; // may need to use other form for attributes
        size_t datasetSize;
        size_t attributeSize;
    };

    enum class CkptDataType {
        EMB_INFO = 0,
        EMB_DATA = 1,
        EMB_HASHMAP = 2,
        DEV_OFFSET = 3,
        EMB_CURR_STAT = 4,
        NDDR_OFFSET = 5,
        NDDR_FEATMAP = 6,
        TENSOR_2_THRESH = 7,
        HIST_REC = 8,
        ATTRIBUTE = 9
    };
} // end namespace MxRec

#define KEY_PROCESS "\033[45m[KeyProcess]\033[0m "
#ifdef GTEST
    #define GTEST_PRIVATE public
#else
    #define GTEST_PRIVATE private
#endif
#endif
