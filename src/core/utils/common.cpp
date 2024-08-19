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

#include "common.h"

#include <memory>
#include <string>
#include <stdexcept>
#include <experimental/filesystem>
#include <unistd.h>
#include <regex>

#include <mpi.h>

#include <dsmi_common_interface.h>
#include <iomanip>

using namespace std;
using std::chrono::system_clock;

namespace MxRec {
    namespace fs = std::experimental::filesystem;

    bool g_isGlogInit = false;
    bool GlogConfig::gStatOn = false;
    int GlogConfig::gGlogLevel;
    string GlogConfig::gRankId;

    ock::ctr::FactoryPtr factory {};

    RankInfo::RankInfo(int rankId, int deviceId, int localRankSize, int option, const vector<int>& ctrlSteps)
        : rankId(rankId), deviceId(deviceId), localRankSize(localRankSize), option(option), ctrlSteps(ctrlSteps)
    {
        MPI_Comm_size(MPI_COMM_WORLD, &rankSize);
        if (localRankSize != 0) {
            localRankId = rankId % localRankSize;
        }
        useStatic = static_cast<unsigned int>(option) bitand HybridOption::USE_STATIC;
        useDynamicExpansion = static_cast<unsigned int>(option) bitand HybridOption::USE_DYNAMIC_EXPANSION;
        useSumSameIdGradients = static_cast<unsigned int>(option) bitand HybridOption::USE_SUM_SAME_ID_GRADIENTS;
    }

    RankInfo::RankInfo(int localRankSize, int option, const vector<int>& maxStep)
        : localRankSize(localRankSize), option(option), ctrlSteps(maxStep)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
        MPI_Comm_size(MPI_COMM_WORLD, &rankSize);
        if (localRankSize != 0) {
            localRankId = rankId % localRankSize;
        }
        useStatic = static_cast<unsigned int>(option) & HybridOption::USE_STATIC;
    }

    RandomInfo::RandomInfo(int start, int len, float constantVal, float randomMin, float randomMax)
        : start(start), len(len), constantVal(constantVal), randomMin(randomMin), randomMax(randomMax)
    {}

    void SetLog(int rank)
    {
        GlogConfig::gGlogLevel = GlobalEnv::glogStderrthreshold;
        if (GlogConfig::gRankId.empty()) {
            GlogConfig::gRankId = std::to_string(rank);
        }
        if (!g_isGlogInit) {
            Logger::SetLevel(GlogConfig::gGlogLevel);
            Logger::SetRank(rank);
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
        // validate file privilege
        fs::perms permissions = fs::status(dataDir).permissions();
        if ((permissions & fs::perms::owner_read) == fs::perms::none) {
            throw invalid_argument(StringFormat("no read permission for file:%s", dataDir.c_str()));
        }
    }

    bool CheckFilePermission(const string& filePath)
    {
        struct stat fileInfo;
        int ret = stat(filePath.c_str(), &fileInfo);
        if (ret != 0) {
            LOG_ERROR("get file {} stat info failed", filePath.c_str());
            return false;
        }

        mode_t mask = 0700;
        mode_t maxMode = 0640;
        const int perPermWidth = 3;
        vector<string> permMsg = { "Other group permission", "Owner group permission", "Owner permission" };
        for (int i = perPermWidth; i > 0; i--) {
            int curPerm = (fileInfo.st_mode & mask) >> ((i - 1) * perPermWidth);
            int maxPerm = (maxMode & mask) >> ((i - 1) * perPermWidth);
            mask = mask >> perPermWidth;
            if (curPerm > maxPerm) {
                LOG_ERROR(" {} : Check {} : Current permission is {}, but required no greater than {} ",
                          filePath.c_str(), permMsg[i - 1], curPerm, maxPerm);
                return false;
            }
        }
        return true;
    }

    std::string FloatPtrToLimitStr(float* ptr, const size_t& prtSize)
    {
        constexpr size_t maxDispLen = 10; // max display number
        int maxLen = static_cast<int>(std::min(prtSize, maxDispLen));
        std::string s;
        for (int i = 0; i < maxLen; i++) {
            s += std::to_string(*(ptr + i)) + " ";
        }
        return s;
    }

    ostream& operator<<(ostream& ss, MxRec::CkptDataType type)
    {
        ss << static_cast<int>(type);
        return ss;
    }

    int GetStepFromPath(const string& loadPath)
    {
        regex pattern(SAVE_SPARSE_PATH_PREFIX + "-.*-(\\d+)");
        smatch match;
        if (!regex_search(loadPath, match, pattern)) {
            return 0;
        }
        int res = 0;
        unsigned int minSize = 2;
        if (match.size() < minSize) {
            return res;
        }
        try {
            res = stoi(match[1]);
        } catch (const std::invalid_argument& e) {
            LOG_ERROR("argument is invalid: {}", e.what());
        }
        return res;
    }

    // Make key for mutex and cv in swap pipeline, id: threadId, channelId: train/eval.
    string MakeSwapCVName(int id, const string& tableName, int channelId)
    {
        return to_string(id) + tableName + to_string(channelId);
    }

    std::string CkptDataTypeName(CkptDataType type)
    {
        switch (type) {
            case CkptDataType::EMB_INFO:
                return "EMB_INFO";
            case CkptDataType::EMB_DATA:
                return "EMB_DATA";
            case CkptDataType::EMB_HASHMAP:
                return "EMB_HASHMAP";
            case CkptDataType::DEV_OFFSET:
                return "DEV_OFFSET";
            case CkptDataType::EMB_CURR_STAT:
                return "EMB_CURR_STAT";
            case CkptDataType::NDDR_OFFSET:
                return "NDDR_OFFSET";
            case CkptDataType::NDDR_FEATMAP:
                return "NDDR_FEATMAP";
            case CkptDataType::TABLE_2_THRESH:
                return "TABLE_2_THRESH";
            case CkptDataType::HIST_REC:
                return "HIST_REC";
            case CkptDataType::ATTRIBUTE:
                return "ATTRIBUTE";
            case CkptDataType::DDR_FREQ_MAP:
                return "DDR_FREQ_MAP";
            case CkptDataType::EXCLUDE_FREQ_MAP:
                return "EXCLUDE_FREQ_MAP";
            case CkptDataType::EVICT_POS:
                return "EVICT_POS";
            case CkptDataType::KEY_COUNT_MAP:
                return "KEY_COUNT_MAP";
            default:
                return "UNKNOWN";
        }
    }
} // end namespace MxRec
