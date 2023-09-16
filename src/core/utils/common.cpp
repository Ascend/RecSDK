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
    string g_rankId;
    int g_glogLevel;
    bool g_isGlogInit = false;
    bool g_statOn = false;

    RankInfo::RankInfo(int rankId, int deviceId, int localRankSize, int option, const vector<int>& maxStep)
        : rankId(rankId), deviceId(deviceId), localRankSize(localRankSize), option(option), maxStep(maxStep)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &rankSize);
        if (localRankSize != 0) {
            localRankId = rankId % localRankSize;
        }
        useStatic = option bitand HybridOption::USE_STATIC;
        useHot = option bitand HybridOption::USE_HOT;
        useDynamicExpansion = option bitand HybridOption::USE_DYNAMIC_EXPANSION;
    }

    RankInfo::RankInfo(int localRankSize, int option, const vector<int>& maxStep)
        : localRankSize(localRankSize), option(option), maxStep(maxStep)
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
            initializer = make_shared<ConstantInitializer>(start, len, constantInitializerInfo.constantValue,
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
            initializer = make_shared<TruncatedNormalInitializer>(start, len, ret);
        } else if (name == "random_normal_initializer") {
            initializerType = InitializerType::RANDOM_NORMAL;
            initializer = make_shared<RandomNormalInitializer>(start, len, ret);
        } else {
            throw std::invalid_argument("Invalid Initializer Type.");
        }
    }

    void SetLog(int rank)
    {
        g_glogLevel = GlobalEnv::glogStderrthreshold;
        if (g_rankId.empty()) {
            g_rankId = std::to_string(rank);
        }
        if (!g_isGlogInit) {
            Log::SetLevel(g_glogLevel);
            Log::SetRank(rank);
            g_isGlogInit = true;
        }
    }

    string GetChipName(int devID)
    {
        int ret = 0;
        struct dsmi_chip_info_stru info = {{ 0 },
                                           { 0 },
                                           { 0 }};
        ret = dsmi_get_chip_info(devID, &info);
        if (ret == 0) {
            stringstream ss;
            ss << info.chip_name;
            LOG_DEBUG("dsmi_get_chip_info successful, ret = {}, chip_name = {}", ret, ss.str());
            return ss.str();
        }

        throw std::runtime_error("dsmi_get_chip_info failed, ret = " + to_string(ret));
    }

    int GetThreadNumEnv()
    {
        return GlobalEnv::keyProcessThreadNum;
    }

    void ValidateReadFile(const string& dataDir, size_t datasetSize)
    {
        // validate soft link
        struct stat fileInfo;
        if (lstat(dataDir.c_str(), &fileInfo) != -1) {
            if (S_ISLNK(fileInfo.st_mode)) {
                LOG_ERROR("soft link {} should not in the path parameter", dataDir);
                throw invalid_argument(StringFormat("soft link should not be the path parameter"));
            }
        }
        // validate file size
        if (datasetSize > FILE_MAX_SIZE) {
            LOG_ERROR("the reading file size is invalid, not in range [{},{}]", FILE_MIN_SIZE, FILE_MAX_SIZE);
            throw invalid_argument(StringFormat("file size invalid"));
        }
    }

    ostream& operator<<(ostream& ss, MxRec::CkptDataType type)
    {
        ss << static_cast<int>(type);
        return ss;
    }

} // end namespace MxRec
