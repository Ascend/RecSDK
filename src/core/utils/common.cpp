/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: common module
 * Author: MindX SDK
 * Create: 2021
 * History: NA
 */

#include "common.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/cfg/env.h>
#include <mpi.h>

#include <dsmi_common_interface.h>

using namespace std;
using std::chrono::system_clock;

namespace MxRec {
    int PerfConfig::keyProcessThreadNum = DEFAULT_KEY_PROCESS_THREAD;
    int PerfConfig::maxUniqueThreadNum = DEFAULT_MAX_UNIQUE_THREAD_NUM;
    bool PerfConfig::fastUnique = false;


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

    ConstantInitializerInfo::ConstantInitializerInfo(float constantValue)
        : constantValue(constantValue)
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
            constantInitializer = ConstantInitializer(start, len, constantInitializerInfo.constantValue);
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
        auto logger = spdlog::stderr_color_mt("console");
        spdlog::set_default_logger(logger);
        std::string pattern = "[%H:%M:%S.%e] [" + std::to_string(rank) + "] [%^%l%$] %v";
        spdlog::default_logger()->set_pattern(pattern);
        auto env_val = spdlog::details::os::getenv("SPDLOG_LEVEL");
        spdlog::cfg::load_env_levels();
    }

    string GetChipName(int devID)
    {
        int ret = 0;
        struct dsmi_chip_info_stru info = {{ 0 },
                                           { 0 },
                                           { 0 }};
        ret = dsmi_get_chip_info(devID, &info);
        if (ret == 0) {
            spdlog::debug("dsmi_get_chip_info successful, ret = {}, chip_name = {}", ret,
                          reinterpret_cast<const char*>(info.chip_name));
            return reinterpret_cast<const char*>(info.chip_name);
        }

        throw std::runtime_error("dsmi_get_chip_info failed, ret = " + to_string(ret));
    }
} // end namespace MxRec
