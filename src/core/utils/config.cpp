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


#include "config.h"
#include "logger.h"

using namespace std;

namespace MxRec {
    // 设置环境变量默认值
    int GlobalEnv::aclTimeout = -1; // 默认阻塞方式，一直等待直到数据接收完成。
    int GlobalEnv::hdChannelSize = 40; // 默认通道深度40
    int GlobalEnv::keyProcessThreadNum = 6; // 默认6个线程
    int GlobalEnv::maxUniqueThreadNum = 8; // 默认最大8个线程
    bool GlobalEnv::fastUnique = false;
    int GlobalEnv::hotEmbUpdateStep = 1000;  // 默认1000步更新
    int GlobalEnv::glogStderrthreshold = 0;  // 默认info级别
    bool GlobalEnv::useCombineFaae = false;
    bool GlobalEnv::recordKeyCount = false; // 默认不打开记录key count的开关
    bool GlobalEnv::useShmSwap = false;     // By default, the switchover of shared memory is disabled
    bool GlobalEnv::hugeTlbEnable = false;  // By default, the large page switch SHM_HUGETLB is not enabled
    int GlobalEnv::ssdSaveCompactLevel = 2;  // 0:完全不压缩；1：只压缩超阈值的文件；2：所有文件都压缩

    /// 配置环境变量，Python侧已经做了变量值校验，CPP侧直接使用即可；bool类型，1代表true，0代表false
    void ConfigGlobalEnv()
    {
        // 设置ACL超时时间
        const char *envAclTimeout = getenv(RecEnvNames::ACL_TIMEOUT);
        if (envAclTimeout != nullptr) {
            GlobalEnv::aclTimeout = std::stoi(envAclTimeout);
        }

        // 设置ACL通道深度
        const char *envHDChannelSize = getenv(RecEnvNames::HD_CHANNEL_SIZE);
        if (envHDChannelSize != nullptr) {
            GlobalEnv::hdChannelSize = std::stoi(envHDChannelSize);
        }

        // 设置数据处理线程数
        const char *envKPNum = getenv(RecEnvNames::KEY_PROCESS_THREAD_NUM);
        if (envKPNum != nullptr) {
            GlobalEnv::keyProcessThreadNum = std::stoi(envKPNum);
        }

        // 设置去重处理线程数
        const char *envUniqNum = getenv(RecEnvNames::MAX_UNIQUE_THREAD_NUM);
        if (envUniqNum != nullptr) {
            GlobalEnv::maxUniqueThreadNum = std::stoi(envUniqNum);
        }

        // 设置是否使用fast unique库进行去重
        const char *envFastUnique = getenv(RecEnvNames::FAST_UNIQUE);
        if (envFastUnique != nullptr) {
            GlobalEnv::fastUnique = (std::stoi(envFastUnique) == 1);
        }

        // 设置hot emb更新步数
        const char *envHotEmbStep = getenv(RecEnvNames::HOT_EMB_UPDATE_STEP);
        if (envHotEmbStep != nullptr) {
            GlobalEnv::hotEmbUpdateStep = std::stoi(envHotEmbStep);
        }

        // 设置日志级别
        const char *envLogLevel = getenv(RecEnvNames::GLOG_STDERR_THRESHOLD);
        if (envLogLevel != nullptr) {
            GlobalEnv::glogStderrthreshold = std::stoi(envLogLevel);
        }

        // 设置特征准入统计模式
        const char *envFAAEMode = getenv(RecEnvNames::USE_COMBINE_FAAE);
        if (envFAAEMode != nullptr) {
            GlobalEnv::useCombineFaae = (std::stoi(envFAAEMode) == 1);
        }

        // 设置打开记录开关，记录batch中key与出现的count的数目
        const char *envRecordKeyCount = getenv(RecEnvNames::RECORD_KEY_COUNT);
        if (envRecordKeyCount != nullptr) {
            GlobalEnv::recordKeyCount = (std::stoi(envRecordKeyCount) == 1);
        }

        // Set the swap in/out switch of shared memory
        const char *envUseShmSwap = getenv(RecEnvNames::USE_SHM_SWAP);
        if (envUseShmSwap != nullptr) {
            GlobalEnv::useShmSwap = (std::stoi(envUseShmSwap) == 1);
        }

        // Enable or disable large pages of shared memory
        const char *envHugeTlbEnable= getenv(RecEnvNames::HUGE_TLB_ENABLE);
        if (envHugeTlbEnable != nullptr) {
            GlobalEnv::hugeTlbEnable = (std::stoi(envHugeTlbEnable) == 1);
        }

        // 设置SSD保存时的压缩等级
        const char *envSsdSaveCompactLevel = getenv(RecEnvNames::SSD_SAVE_COMPACT_LEVEL);
        if (envSsdSaveCompactLevel != nullptr) {
            GlobalEnv::ssdSaveCompactLevel = std::stoi(envSsdSaveCompactLevel);
        }
    }

    void LogGlobalEnv()
    {
        LOG_DEBUG("Environment variables are: [{}: {}], [{}: {}], [{}: {}], [{}: {}], [{}: {}], "
                  "[{}: {}], [{}: {}], [{}: {}], [{}: {}], [{}: {}], [{}: {}].",
                  RecEnvNames::ACL_TIMEOUT, GlobalEnv::aclTimeout,
                  RecEnvNames::HD_CHANNEL_SIZE, GlobalEnv::hdChannelSize,
                  RecEnvNames::KEY_PROCESS_THREAD_NUM, GlobalEnv::keyProcessThreadNum,
                  RecEnvNames::MAX_UNIQUE_THREAD_NUM, GlobalEnv::maxUniqueThreadNum,
                  RecEnvNames::FAST_UNIQUE, GlobalEnv::fastUnique,
                  RecEnvNames::HOT_EMB_UPDATE_STEP, GlobalEnv::hotEmbUpdateStep,
                  RecEnvNames::GLOG_STDERR_THRESHOLD, GlobalEnv::glogStderrthreshold,
                  RecEnvNames::USE_COMBINE_FAAE, GlobalEnv::useCombineFaae,
                  RecEnvNames::RECORD_KEY_COUNT, GlobalEnv::recordKeyCount,
                  RecEnvNames::USE_SHM_SWAP, GlobalEnv::useShmSwap,
                  RecEnvNames::HUGE_TLB_ENABLE, GlobalEnv::hugeTlbEnable,
                  RecEnvNames::SSD_SAVE_COMPACT_LEVEL, GlobalEnv::ssdSaveCompactLevel);
    }
}
