/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: operator module
 * Author: MindX SDK
 * Date: 2022/11/23
 */

#ifndef FEATURE_ADMIT_AND_EVICT_H
#define FEATURE_ADMIT_AND_EVICT_H

#include <cstdint>
#include <vector>
#include <thread>
#include <unordered_map>
#include <queue>
#include <memory>
#include <mutex>
#include <string>
#include "absl/container/flat_hash_map.h"
#include "host_emb/host_emb.h"
#include "utils/common.h"
#include "utils/safe_queue.h"
#include "utils/singleton.h"
#include "utils/time_cost.h"

namespace MxRec {
    enum class FeatureAdmitType {
        FEATURE_ADMIT_OK = 0,
        FEATURE_ADMIT_FAILED
    };
    enum class FeatureAdmitReturnType {
        FEATURE_ADMIT_RETURN_OK = 0,
        FEATURE_ADMIT_RETURN_ERROR
    };

    enum class SingleEmbTableStatus {
        SETS_BOTH = 0,         // 准入&淘汰功能，正常（threshold配置正常 && 传batch时间戳）
        SETS_NONE,             // 没有配置（不支持）
        SETS_ONLY_ADMIT,       // 只支持准入功能
        SETS_ERROR             // 只有淘汰、没有准入的错误，或其它
    };

    const int DEFAULT_RECORDS_INIT_SIZE = 10000;
    const int FEATURE_EVICT_TIME_INTERVAL = 3600 * 24;
    const int FEATURE_MIN_TIME_INTERVAL = 10;

    class FeatureAdmitAndEvict {
    public:
        explicit FeatureAdmitAndEvict(int recordsInitSize = DEFAULT_RECORDS_INIT_SIZE);
        ~FeatureAdmitAndEvict();

        bool Init(const std::vector<ThresholdValue>& thresholdValues);

        // 以下为类的公共接口
        // 特征准入接口
        FeatureAdmitReturnType FeatureAdmit(int channel, const std::unique_ptr<EmbBatchT>& batch,
            KeysT& splitKey, std::vector<uint32_t>& keyCount);

        // 特征淘汰接口
        void FeatureEvict(map<std::string, std::vector<emb_key_t>>& evictKeyMap);
        void ExecuteFeatureAdmit(
            const string& tableName, int channel, KeysT& splitKey, absl::flat_hash_map<int64_t, uint32_t>& mergeKeys);

        // 特征淘汰的使能接口
        void SetFunctionSwitch(bool isEnableEvict);
        // 特征准入合表统计
        void SetCombineSwitch();

        bool GetFunctionSwitch() const;
        void PreProcessKeys(const std::vector<int64_t>& splitKey, std::vector<uint32_t>& keyCount,
            absl::flat_hash_map<int64_t, uint32_t>& mergeKeys) const;

        // 判断配置是否正确的接口
        static bool IsThresholdCfgOK(const std::vector<ThresholdValue>& thresholds,
            const std::vector<std::string>& embNames, bool isTimestamp);

        bool SetTableThresholds(int threshold, string embName);
        bool SetTableThreshold(int threshold, string embName);
        // 与模型保存加载交互的接口
        auto GetTableThresholds() -> Table2ThreshMemT;
        auto GetHistoryRecords() -> AdmitAndEvictData&;

        void LoadTableThresholds(Table2ThreshMemT& loadData);
        void LoadHistoryRecords(AdmitAndEvictData& loadData);

        static std::vector<ThresholdValue> m_cfgThresholds;                        // 用于判断阈值配置的有效性
        static absl::flat_hash_map<std::string, SingleEmbTableStatus> m_embStatus; // 用于“准入&淘汰”功能解耦

        GTEST_PRIVATE :

        // 解析m_table2Threshold
        bool ParseThresholdCfg(const std::vector<ThresholdValue>& thresholdValues);
        std::vector<std::string> GetAllNeedEvictTableNames();
        FeatureAdmitType FeatureAdmitHelper(const int channel, const std::string& tableNameOrigin,
                                            const int64_t featureId, const uint32_t featureCnt);
        void FeatureEvictHelper(const std::string& embName, std::vector<emb_key_t>& evictKey);
        void ResetAllRecords();

        bool m_isEnableFunction { true };                                    // “特征淘汰”的使能开关
        bool m_isExit { false };                                             // 淘汰线程退出的标识
        bool m_isCombine { false };                                          // 是否合并统计history
        absl::flat_hash_map<std::string, ThresholdValue> m_table2Threshold; // table-X ---> ThresholdValue 映射
        AdmitAndEvictData m_recordsData;
        std::mutex m_syncMutexs;         // 特征准入与特征淘汰竞争的同步锁
        int m_recordsInitSize { DEFAULT_RECORDS_INIT_SIZE }; // m_historyRecords表初始容量
        std::thread m_evictThread; // 特征淘汰功能，以“线程 + 定时任务”方式实现
    };
}

#endif // FEATURE_ADMIT_AND_EVICT_H
