/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#ifndef COMMON_H
#define COMMON_H

#include <sys/stat.h>
#include <cstring>
#include <vector>
#include <random>
#include <chrono>
#include <map>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "tensorflow/core/framework/tensor.h"
#include "absl/container/flat_hash_map.h"
#include "securec.h"
#include "utils/logger.h"
#include "utils/config.h"

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
    constexpr char SUM_SAME_ID[] = "sum_same_id_gradients_and_apply";
    constexpr size_t MAX_VOCABULARY_SIZE = 1e10;
    constexpr int SSD_SIZE_INDEX = 2;
    constexpr int MAX_FILE_NUM = 1000;
    // for GLOG
    struct GlogConfig {
        static bool gStatOn;
        static int gGlogLevel;
        static string gRankId;
    };

    constexpr int GLOG_MAX_BUF_SIZE = 1024;
    constexpr int GLOG_TIME_WIDTH_2 = 2;
    constexpr int GLOG_TIME_WIDTH_6 = 6;
    constexpr char GLOG_STAT_FLAG[] = "statOn";

    // unique related config
    constexpr int UNIQUE_BUCKET = 6;
    constexpr int MIN_UNIQUE_THREAD_NUM = 1;

    // validate file
    constexpr long long FILE_MAX_SIZE = 1LL << 40;
    constexpr int FILE_MIN_SIZE = 0;
    constexpr size_t BUFFER_SIZE{1024 * 1024 * 64};
    constexpr size_t MAP_BYTE_SIZE{static_cast<size_t>(10) * 1024 * 1024 * 1024};
#ifdef GTEST
    constexpr int KEY_PROCESS_TIMEOUT = 3;
#else
    constexpr int KEY_PROCESS_TIMEOUT = 120;
#endif
    constexpr int GET_BATCH_TIMEOUT = 300;
    constexpr int EOS_TIMEOUT = 30;

    constexpr size_t DEFAULT_RANDOM_SEED = 10086;
    // constexpr int INVALID_KEY_VALUE = -1;
    constexpr int64_t INVALID_KEY_VALUE = -1;
    constexpr int ALLTOALLVC_ALIGN = 128;
    constexpr int PROFILING_START_BATCH_ID = 100;
    constexpr int PROFILING_END_BATCH_ID = 200;
    constexpr int MGMT_THREAD_BIND = 48;
    constexpr int UNIQUE_MAX_BUCKET_WIDTH = 6;
    constexpr int HOT_EMB_UPDATE_STEP_DEFAULT = 1000;
    constexpr float HOT_EMB_CACHE_PCT = static_cast<float>(1. / 3);  // hot emb cache percent

    const string COMBINE_HISTORY_NAME = "combine_table_history";

    using emb_key_t = int64_t;
    using freq_num_t = int64_t;
    using EmbNameT= std::string;
    using KeysT = std::vector<emb_key_t>;
    using LookupKeyT = std::tuple<int, EmbNameT, KeysT>;             // batch_id quarry_lable keys_vector
    using TensorInfoT = std::tuple<int, EmbNameT, std::list<std::unique_ptr<std::vector<Tensor>>>::iterator>;

    namespace HybridOption {
        const unsigned int USE_STATIC = 0x001;
        const unsigned int USE_DYNAMIC_EXPANSION = 0x001 << 1;
    };

    string GetChipName(int devID);
    int GetThreadNumEnv();

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
        const int ASCEND910_B2C = 196608;
    };

    inline int GetUBSize(int devID)
    {
        const std::map<string, int> chipUbSizeList = {{"910A", UBSize::ASCEND910_A},
            {"910B", UBSize::ASCEND910_B},
            {"920A", UBSize::ASCEND920_A},
            {"910B1", UBSize::ASCEND910_B1},
            {"910B2", UBSize::ASCEND910_B2},
            {"910B3", UBSize::ASCEND910_B3},
            {"910B4", UBSize::ASCEND910_B4},
            {"910B2C", UBSize::ASCEND910_B2C},
            {"910C1", UBSize::ASCEND910_C1},
            {"910C2", UBSize::ASCEND910_C1},
            {"910C3", UBSize::ASCEND910_C3}
        };
        auto it = chipUbSizeList.find(GetChipName(devID));
        if (it != chipUbSizeList.end()) {
            return it->second;
        }

        throw std::runtime_error("unknown chip ub size" + GetChipName(devID));
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
            constexpr size_t maxDispLen = 20;
            int maxLen = static_cast<int>(std::min(sample.size(), maxDispLen));
            for (int i = 0; i < maxLen; i++) {
                s += std::to_string(sample[i]) + " ";
            }
            return s;
        }

        std::vector<T> sample;
        std::string name;
        size_t batchSize;
        int batchId;
        int channel = 0;
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
        const void *tensor;
    };

    using EmbBatchT = Batch<int64_t>;
    using BatchTaskT = BatchTask;

    struct DDRParam {
        vector<Tensor> tmpDataOut;
        vector<int32_t> offsetsOut;
        DDRParam(vector<Tensor> tmpData, vector<int32_t> offset)
        {
            tmpDataOut = tmpData;
            offsetsOut = offset;
        }
    };

    struct RankInfo {
        RankInfo() = default;

        RankInfo(int rankId, int deviceId, int localRankSize, int option, const std::vector<int>& ctrlSteps);
        RankInfo(int localRankSize, int option, const std::vector<int>& maxStep);

        int rankId {};
        int deviceId {};
        int rankSize {};
        int localRankId {};
        int localRankSize {};
        bool useStatic { false };
        uint32_t option {};
        int nBatch {};
        bool isDDR { false };
        bool isSSDEnabled { false };
        bool useDynamicExpansion {false};
        std::vector<int> ctrlSteps; // 包含三个步数: train_steps, eval_steps, save_steps
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

    struct EmbeddingSizeInfo {
        EmbeddingSizeInfo() = default;
        EmbeddingSizeInfo(size_t embSize, size_t extendSize)
        {
            embeddingSize = embSize;
            extendEmbSize = extendSize;
        }

        size_t embeddingSize;
        size_t extendEmbSize;
    };

    struct OptimizerInfo {
        OptimizerInfo() = default;
        OptimizerInfo(std::string name, vector<std::string> params)
        {
            optimName = name;
            optimParams = std::move(params);
        }

        std::string optimName;
        vector<std::string> optimParams;
    };

    struct ThresholdValue {
        ThresholdValue() = default;
        ThresholdValue(EmbNameT name, int countThre, int timeThre, int faaeCoef, bool isSum)
        {
            tableName = name;
            countThreshold = countThre;
            timeThreshold = timeThre;
            faaeCoefficient = faaeCoef;
            isEnableSum = isSum;
        }

        EmbNameT tableName { "" }; // embName
        int countThreshold { -1 }; // 只配置count，即“只有准入、而没有淘汰”功能，对应SingleHostEmbTableStatus::SETS_ONLY_ADMIT状态
        int timeThreshold { -1 };  // 只配置time，配置错误；即准入是淘汰的前提，对应SingleHostEmbTableStatus::SETS_BOTH状态
        int faaeCoefficient { 1 }; // 配置后,该表在准入时，count计数会乘以该系数
        bool isEnableSum {true};   // 配置false,该表在准入时，count计数不会累加
    };

    struct FeatureItemInfo {
        FeatureItemInfo() = default;
        FeatureItemInfo(uint32_t cnt, time_t lastT)
            : count(cnt), lastTime(lastT)
        {}

        uint32_t count { 0 };
        time_t lastTime { 0 };
    };

    using HistoryRecords = absl::flat_hash_map<std::string, absl::flat_hash_map<int64_t, FeatureItemInfo>>;
    struct AdmitAndEvictData {
        HistoryRecords historyRecords;                       // embName ---> {id, FeatureItemInfo} 映射
        absl::flat_hash_map<std::string, time_t> timestamps; // 用于特征准入&淘汰的时间戳
    };

    void SetLog(int rank);

    template<typename ... Args>
    string StringFormat(const string& format, Args ... args)
    {
        auto size = static_cast<size_t>(GLOG_MAX_BUF_SIZE);
        auto buf = std::make_unique<char[]>(size);
        memset_s(buf.get(), size, 0, size);
        int nChar =  snprintf_s(buf.get(), size, size - 1, format.c_str(), args ...);
        if (nChar == -1) {
            throw invalid_argument("StringFormat failed");
        }
        return string(buf.get(), buf.get() + nChar);
    }

    // use environment variable GLOG_v to decide if showing debug log.
    // default 0, debug message will not display.
    // 1 for debug, 2 for trace
    constexpr int GLOG_DEBUG = 1;
    constexpr int GLOG_TRACE = 2;

    template<typename T>
    std::string VectorToString(const std::vector<T>& vec)
    {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < vec.size(); ++i) {
            ss << vec[i];
            if (i != vec.size() - 1) {
                ss << ", ";
            }
        }
        ss << "]";
        return ss.str();
    }

    template<typename K, typename V>
    std::string MapToString(const std::map<K, V>& map)
    {
        std::stringstream ss;
        ss << "{";
        for (auto it = map.begin(); it != map.end(); ++it) {
            ss << it->first << ": " << it->second;
            if (std::next(it) != map.end()) {
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();
    }

    template<typename K, typename V>
    std::string MapToString(const absl::flat_hash_map<K, V>& map)
    {
        std::stringstream ss;
        ss << "{";
        for (auto it = map.begin(); it != map.end(); ++it) {
            ss << it->first << ": " << it->second;
            if (std::next(it) != map.end()) {
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();
    }

    void ValidateReadFile(const string& dataDir, size_t datasetSize);

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

    struct EmbInfoParams {
        EmbInfoParams() = default;

        EmbInfoParams(const std::string& name,
                int sendCount,
                int embeddingSize,
                int extEmbeddingSize,
                bool isSave,
                bool isGrad)
            : name(name),
              sendCount(sendCount),
              embeddingSize(embeddingSize),
              extEmbeddingSize(extEmbeddingSize),
              isSave(isSave),
              isGrad(isGrad)
        {
        }
        std::string name;
        int sendCount;
        int embeddingSize;
        int extEmbeddingSize;
        bool isSave;
        bool isGrad;
    };

    struct EmbInfo {
        EmbInfo() = default;

        EmbInfo(const EmbInfoParams& embInfoParams,
                std::vector<size_t> vocabsize,
                std::vector<InitializeInfo> initializeInfos,
                std::vector<std::string> ssdDataPath)
            : name(embInfoParams.name),
              sendCount(embInfoParams.sendCount),
              embeddingSize(embInfoParams.embeddingSize),
              extEmbeddingSize(embInfoParams.extEmbeddingSize),
              isSave(embInfoParams.isSave),
              isGrad(embInfoParams.isGrad),
              devVocabSize(vocabsize[0]),
              hostVocabSize(vocabsize[1]),
              ssdVocabSize(vocabsize[SSD_SIZE_INDEX]),
              initializeInfos(initializeInfos),
              ssdDataPath(std::move(ssdDataPath))
        {
        }

        std::string name;
        int sendCount;
        int embeddingSize;
        int extEmbeddingSize;
        bool isSave;
        bool isGrad;
        size_t devVocabSize;
        size_t hostVocabSize;
        size_t ssdVocabSize;
        std::vector<InitializeInfo> initializeInfos;
        std::vector<std::string> ssdDataPath;
    };

    struct HostEmbTable {
        EmbInfo hostEmbInfo;
        std::vector<std::vector<float>> embData;
    };

    struct EmbHashMapInfo {
        absl::flat_hash_map<emb_key_t, int64_t> hostHashMap; // key在HBM中的偏移
        std::vector<int> devOffset2Batch; // has -1
        std::vector<emb_key_t> devOffset2Key;
        size_t currentUpdatePos;
        size_t currentUpdatePosStart;
        size_t hostVocabSize;
        size_t devVocabSize;
        size_t freeSize;
        std::vector<int32_t> lookUpVec;
        std::vector<size_t> missingKeysHostPos; // 用于记录当前batch在host上需要换出的偏移
        std::vector<size_t> swapPos; // 记录从HBM换出到DDR的offset
        /*
         * 取值范围：[0,devVocabSize+hostVocabSize);
         * [0,devVocabSize-1]时存储在HBM, [devVocabSize,devVocabSize+hostVocabSize)存储在DDR
         */
        size_t maxOffset { 0 };
        /*
         * 记录DDR内淘汰列表，其值为相对HBM+DDR大表的；hostHashMap可直接使用；操作ddr内emb时需减掉devVocabSize
         * 例如：HBM表大小20(offset:0~19)，DDR表大小为100（offset:0~99）；
         * 若DDR内0位置被淘汰，记录到evictPos的值为0+20=20
         */
        std::vector<size_t> evictPos;
        std::vector<size_t> evictDevPos; // 记录HBM内淘汰列表
        size_t maxOffsetOld { 0 };
        std::vector<size_t> evictPosChange;
        std::vector<size_t> evictDevPosChange;
        std::vector<std::pair<int, emb_key_t>> devOffset2KeyOld;
        std::vector<std::pair<emb_key_t, emb_key_t>> oldSwap; // (old on dev, old on host)
        /*
         * HBM与DDR换入换出时,已存在于DDR且要转移到HBM的key(不包含新key); 用于SSD模式
         * (区别于oldSwap: pair.second为已存在于DDR key + 换入换出前映射到DDR的新key)
         */
        std::vector<emb_key_t> ddr2HbmKeys;
        void SetStartCount();

        bool HasFree(size_t i) const;
    };

    struct All2AllInfo {
        KeysT keyRecv;
        vector<int> scAll;
        vector<uint32_t> countRecv;
        All2AllInfo() = default;
        All2AllInfo(KeysT keyRecv, vector<int> scAll, vector<uint32_t> countRecv)
            : keyRecv(keyRecv), scAll(scAll), countRecv(countRecv) {}
    };

    struct UniqueInfo {
        vector<int32_t> restore;
        vector<int32_t> hotPos;
        All2AllInfo all2AllInfo;
        UniqueInfo() = default;
        UniqueInfo(vector<int32_t> restore, vector<int32_t> hotPos, All2AllInfo all2AllInfo)
            : restore(restore), hotPos(hotPos), all2AllInfo(all2AllInfo) {}
    };

    struct KeySendInfo {
        KeysT keySend;
        vector<int32_t> keyCount;
    };

    using EmbMemT = absl::flat_hash_map<std::string, HostEmbTable>;
    using EmbHashMemT = absl::flat_hash_map<std::string, EmbHashMapInfo>;
    using OffsetMemT = std::map<EmbNameT, size_t>;
    using KeyOffsetMemT = std::map<EmbNameT, absl::flat_hash_map<emb_key_t, int64_t>>;
    using KeyCountMemT = std::map<EmbNameT, absl::flat_hash_map<emb_key_t, size_t>>;
    using Table2ThreshMemT = absl::flat_hash_map<std::string, ThresholdValue>;
    using trans_serialize_t = uint8_t;
    using OffsetMapT = std::map<EmbNameT, std::vector<int64_t>>;
    using OffsetT = std::vector<int64_t>;
    using AllKeyOffsetMapT = std::map<std::string, std::map<int64_t, int64_t>>;
    using KeyFreqMemT = unordered_map<std::string, unordered_map<emb_key_t, freq_num_t>>;

    enum class CkptFeatureType {
        HOST_EMB = 0,
        EMB_HASHMAP = 1,
        MAX_OFFSET = 2,
        KEY_OFFSET_MAP = 3,
        FEAT_ADMIT_N_EVICT = 4,
        DDR_KEY_FREQ_MAP = 5,
        EXCLUDE_DDR_KEY_FREQ_MAP = 6,
        KEY_COUNT_MAP = 7
    };

    struct CkptData {
        EmbMemT* hostEmbs = nullptr;
        EmbHashMemT embHashMaps;
        OffsetMemT maxOffset;
        KeyOffsetMemT keyOffsetMap;
        OffsetMapT offsetMap;
        OffsetMapT* offsetMapPtr = &offsetMap;
        KeyCountMemT keyCountMap;
        Table2ThreshMemT table2Thresh;
        AdmitAndEvictData histRec;
        KeyFreqMemT ddrKeyFreqMaps;
        KeyFreqMemT excludeDDRKeyFreqMaps;
    };

    struct CkptTransData {
        std::vector<int64_t> int64Arr;
        std::vector<int64_t> addressArr;
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
        TABLE_2_THRESH = 7,
        HIST_REC = 8,
        ATTRIBUTE = 9,
        DDR_FREQ_MAP = 10,
        EXCLUDE_FREQ_MAP = 11,
        EVICT_POS = 12,
        KEY_COUNT_MAP = 13
    };

    ostream& operator<<(ostream& ss, MxRec::CkptDataType type);
    bool CheckFilePermission(const string& filePath);
} // end namespace MxRec

#define KEY_PROCESS "\033[45m[KeyProcess]\033[0m "
#define STAT_INFO "[StatInfo] "
#ifdef GTEST
    #define GTEST_PRIVATE public
#else
    #define GTEST_PRIVATE private
#endif
#endif
