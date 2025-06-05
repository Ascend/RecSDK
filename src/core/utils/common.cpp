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
                auto error = Error(ModuleName::M_UTILS, ErrorType::INVALID_ARGUMENT,
                                   StringFormat("Found soft link in path:%s.", dataDir.c_str()));
                LOG_ERROR(error.ToString());
                throw invalid_argument(error.ToString());
            }
        }
        // validate file size
        if (datasetSize > FILE_MAX_SIZE) {
            auto error = Error(ModuleName::M_UTILS, ErrorType::INVALID_ARGUMENT,
                               StringFormat("The reading file size is invalid, not in range [%d, %d], path:%s.",
                                            FILE_MIN_SIZE, FILE_MAX_SIZE, dataDir.c_str()));
            LOG_ERROR(error.ToString());
            throw invalid_argument(error.ToString());
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
            auto error = Error(ModuleName::M_UTILS, ErrorType::UNKNOWN,
                               StringFormat("Get stat info failed, file:%s, error:%d.", filePath.c_str(), ret));
            LOG_ERROR(error.ToString());
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
                auto error = Error(ModuleName::M_UTILS, ErrorType::INVALID_ARGUMENT,
                                   StringFormat("File permission wrong, file:%s, type:%s, current permission:%d,"
                                                " required no greater than:%d.",
                                                filePath.c_str(), permMsg[i - 1], curPerm, maxPerm));
                LOG_ERROR(error.ToString());
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
            LOG_WARN("Fail to get step from path:{}, error:{}. Set step as 0.", loadPath, e.what());
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
        static const unordered_map<CkptDataType, string> typeNameMap = {
            {CkptDataType::EMB_INFO, "EMB_INFO"},
            {CkptDataType::EMB_DATA, "EMB_DATA"},
            {CkptDataType::EMB_HASHMAP, "EMB_HASHMAP"},
            {CkptDataType::DEV_OFFSET, "DEV_OFFSET"},
            {CkptDataType::EMB_CURR_STAT, "EMB_CURR_STAT"},
            {CkptDataType::NDDR_OFFSET, "NDDR_OFFSET"},
            {CkptDataType::NDDR_FEATMAP, "NDDR_FEATMAP"},
            {CkptDataType::TABLE_2_THRESH, "TABLE_2_THRESH"},
            {CkptDataType::HIST_REC, "HIST_REC"},
            {CkptDataType::ATTRIBUTE, "ATTRIBUTE"},
            {CkptDataType::DDR_FREQ_MAP, "DDR_FREQ_MAP"},
            {CkptDataType::EXCLUDE_FREQ_MAP, "EXCLUDE_FREQ_MAP"},
            {CkptDataType::EVICT_POS, "EVICT_POS"},
            {CkptDataType::KEY_COUNT_MAP, "KEY_COUNT_MAP"}
        };

        auto it = typeNameMap.find(type);
        if (it != typeNameMap.end()) {
            return it->second;
        }

        return "UNKNOWN";
    }

    bool CheckFileExist(const string& filePath)
    {
        std::ifstream file(filePath.c_str());
        return file.is_open();
    }

    void RenameFilePath(const string& filePath, const string& newFilePath)
    {
        if (access(newFilePath.c_str(), F_OK) == 0) {
            LOG_INFO("Target file already exists: {}", newFilePath);
            return;
        }
        if (access(filePath.c_str(), F_OK) != 0) {
            auto error = Error(ModuleName::M_UTILS, ErrorType::INVALID_ARGUMENT,
                               StringFormat("File does not exist:%s.", filePath.c_str()));
            LOG_ERROR(error.ToString());
            throw invalid_argument(error.ToString());
        } else {
            if (rename(filePath.c_str(), newFilePath.c_str()) == 0) {
                LOG_INFO("File renamed successfully: {}", newFilePath);
            } else {
                auto error = Error(ModuleName::M_UTILS, ErrorType::IO_ERROR,
                                   StringFormat("Failed to rename %s to %s.",
                                                filePath.c_str(), newFilePath.c_str()));
                LOG_ERROR(error.ToString());
                throw std::runtime_error(error.ToString());
            }
        }
    }
} // end namespace MxRec
