/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Create: 2021
 * History: NA
 */

#include "common.h"

#include <memory>
#include <string>
#include <stdexcept>

#include <mpi.h>

#include <dsmi_common_interface.h>
#include <iomanip>

using namespace std;
using std::chrono::system_clock;

namespace MxRec {
    int PerfConfig::keyProcessThreadNum = DEFAULT_KEY_PROCESS_THREAD;
    int PerfConfig::maxUniqueThreadNum = DEFAULT_MAX_UNIQUE_THREAD_NUM;
    bool PerfConfig::fastUnique = false;
    bool PerfConfig::gradientStrategy = false;
    string g_rankId;
    int g_glogLevel;
    bool g_isGlogInit = false;


    RankInfo::RankInfo(int rankId, int deviceId, int localRankSize, int option, int nBatch,
        const vector<int>& maxStep) : rankId(rankId), deviceId(deviceId), localRankSize(localRankSize), option(option),
        nBatch(nBatch), maxStep(maxStep)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &rankSize);
        if (localRankSize != 0) {
            localRankId = rankId % localRankSize;
        }
        useStatic = option bitand HybridOption::USE_STATIC;
        useHot = option bitand HybridOption::USE_HOT;
        useDynamicExpansion = option bitand HybridOption::USE_DYNAMIC_EXPANSION;
    }

    RankInfo::RankInfo(int localRankSize, int option, int nBatch, const vector<int>& maxStep)
        : localRankSize(localRankSize), option(option), nBatch(nBatch), maxStep(maxStep)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
        MPI_Comm_size(MPI_COMM_WORLD, &rankSize);
        if (localRankSize != 0) {
            localRankId = rankId % localRankSize;
        }
        useStatic = option & HybridOption::USE_STATIC;
        useHot = option & HybridOption::USE_HOT;
    }

    RandomInfo::RandomInfo(int start, int len, float constantVal, float randomMin, float randomMax)
        : start(start), len(len), constantVal(constantVal), randomMin(randomMin), randomMax(randomMax)
    {}

    ConstantInitializerInfo::ConstantInitializerInfo(float constantValue, float initK)
        : constantValue(constantValue), initK(initK)
    {}

    NormalInitializerInfo::NormalInitializerInfo(float mean, float stddev, int seed, float initK)
        : mean(mean), stddev(stddev), seed(seed), initK(initK)
    {}

    InitializeInfo::InitializeInfo(std::string& name, int start, int len,
        ConstantInitializerInfo constantInitializerInfo)
        : name(name), start(start), len(len), constantInitializerInfo(constantInitializerInfo)
    {
        if (name == "constant_initializer") {
            initializerType = InitializerType::CONSTANT;
            constantInitializer = ConstantInitializer(start, len, constantInitializerInfo.constantValue,
                                                      constantInitializerInfo.initK);
        } else {
            throw std::invalid_argument("Invalid Initializer Type.");
        }
    }

    InitializeInfo::InitializeInfo(std::string& name, int start, int len, NormalInitializerInfo normalInitializerInfo)
        : name(name), start(start), len(len), normalInitializerInfo(normalInitializerInfo)
    {
        std::tuple<float, float, int, float> ret(normalInitializerInfo.mean, normalInitializerInfo.stddev,
                                                 normalInitializerInfo.seed, normalInitializerInfo.initK);

        if (name == "truncated_normal_initializer") {
            initializerType = InitializerType::TRUNCATED_NORMAL;
            truncatedNormalInitializer = TruncatedNormalInitializer(start, len, ret);
        } else if (name == "random_normal_initializer") {
            initializerType = InitializerType::RANDOM_NORMAL;
            randomNormalInitializer = RandomNormalInitializer(start, len, ret);
        } else {
            throw std::invalid_argument("Invalid Initializer Type.");
        }
    }

    void SetLog(int rank)
    {
        auto logLevel = getenv("GLOG_stderrthreshold");
        if (logLevel == nullptr) {
            g_glogLevel = 0;  // default as INFO
        } else {
            g_glogLevel = atoi(logLevel);
        }
        if (g_rankId.empty()) {
            g_rankId = std::to_string(rank);
        }
        if (!g_isGlogInit) {
            InitGoogleLogging("mxRec", &CustomGlogFormat);
            g_isGlogInit = true;
        }
    }

    void CustomGlogFormat(std::ostream &s, const google::LogMessageInfo &l, void*)
    {
        s << "["
          << setw(GLOG_TIME_WIDTH_2) << l.time.hour() << ':'
          << setw(GLOG_TIME_WIDTH_2) << l.time.min()  << ':'
          << setw(GLOG_TIME_WIDTH_2) << l.time.sec() << "."
          << setw(GLOG_TIME_WIDTH_6) << l.time.usec() << "]"
          << " [" + g_rankId + "]"
          << " [" <<  l.severity << "] ";
    }

    string GetChipName(int devID)
    {
        int ret = 0;
        struct dsmi_chip_info_stru info = {{ 0 },
                                           { 0 },
                                           { 0 }};
        ret = dsmi_get_chip_info(devID, &info);
        if (ret == 0) {
            VLOG(GLOG_DEBUG) << StringFormat(
                "dsmi_get_chip_info successful, ret = %d, chip_name = %s", ret,
                reinterpret_cast<const char*>(info.chip_name)
            );
            return reinterpret_cast<const char*>(info.chip_name);
        }

        throw std::runtime_error("dsmi_get_chip_info failed, ret = " + to_string(ret));
    }

    bool GetCombineSwitch()
    {
        const char* faaeMode = std::getenv("USE_COMBINE_FAAE"); // 获取环境变量
        bool isCombine = false;
        if (faaeMode != nullptr) {
            try {
                isCombine = (std::stoi(faaeMode) == 1);
                LOG(INFO) << StringFormat("If combine history table： %d", isCombine);
            } catch (const std::invalid_argument& e) {
                LOG(ERROR) << "The value of USE_COMBINE_FAAE is invalid!";
                throw std::invalid_argument("Invalid env value USE_COMBINE_FAAE");
            }
        }
        return isCombine;
    }

    int GetThreadNumEnv()
    {
        int threadNum = 0;
        const char* threadNumEnv = getenv("KEY_PROCESS_THREAD_NUM");
        if (threadNumEnv != nullptr) {
            try {
                threadNum = std::stoi(threadNumEnv);
            } catch (const std::invalid_argument& e) {
                threadNum = KEY_PROCESS_THREAD;
                LOG(INFO) << StringFormat("error value of threadNum, use default KEY_PROCESS_THREAD: %d",
                                          threadNum);
            }
            if (threadNum > KEY_PROCESS_THREAD || threadNum < 0) {
                throw runtime_error(StringFormat("%d is not valid", threadNum));
            }
        } else {
            threadNum = KEY_PROCESS_THREAD;
            LOG(INFO) << StringFormat("use default KEY_PROCESS_THREAD: %d", threadNum);
        }
        return threadNum;
    }
} // end namespace MxRec
