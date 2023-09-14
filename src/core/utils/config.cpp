/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: config module
 * Author: MindX SDK
 * Create: 2023
 * History: NA
 */


#include "config.h"
#include "log.h"

using namespace std;

namespace MxRec {
    // 设置环境变量默认值
    string GlobalEnv::applyGradientsStrategy = ApplyGradientsStrategyOptions::DIRECT_APPLY;
    int GlobalEnv::aclTimeout = -1; // 默认阻塞方式，一直等待直到数据接收完成。
    int GlobalEnv::hdChannelSize = 40; // 默认通道深度40
    bool GlobalEnv::findOffsetV2 = false;
    bool GlobalEnv::findOffsetV3 = false;
    int GlobalEnv::keyProcessThreadNum = 6; // 默认6个线程
    int GlobalEnv::maxUniqueThreadNum = 8; // 默认最大8个线程
    bool GlobalEnv::fastUnique = false;
    bool GlobalEnv::updateEmbV2 = false;
    int GlobalEnv::hotEmbUpdateStep = 1000;  // 默认1000步更新
    int GlobalEnv::glogStderrthreshold = 0;  // 默认info级别
    bool GlobalEnv::useCombineFaae = false;
    bool GlobalEnv::statOn = false;

    /// 配置环境变量，Python侧已经做了变量值校验，CPP侧直接使用即可；bool类型，1代表true，0代表false
    void ConfigGlobalEnv()
    {
        // 设置梯度策略
        const char *envStrategy = getenv(RecEnvNames::APPLY_GRADIENTS_STRATEGY);
        if (envStrategy != nullptr) {
            GlobalEnv::applyGradientsStrategy = envStrategy;
        }

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

        // 设置偏移查找策略V2
        const char *envFindOffsetV2 = getenv(RecEnvNames::FIND_OFFSET_V2);
        if (envFindOffsetV2 != nullptr) {
            GlobalEnv::findOffsetV2 = (std::stoi(envFindOffsetV2) == 1);
        }

        // 设置偏移查找策略V3
        const char *envFindOffsetV3 = getenv(RecEnvNames::FIND_OFFSET_V3);
        if (envFindOffsetV3 != nullptr) {
            GlobalEnv::findOffsetV3 = (std::stoi(envFindOffsetV3) == 1);
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

        // 设置是否使用异步更新d2h的host emb
        const char *envUpdateEmbV2 = getenv(RecEnvNames::UPDATE_EMB_V2);
        if (envUpdateEmbV2 != nullptr) {
            GlobalEnv::updateEmbV2 = (std::stoi(envUpdateEmbV2) == 1);
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

        // 设置打开维测信息
        const char *envStat = getenv(RecEnvNames::STAT_ON);
        if (envStat != nullptr) {
            GlobalEnv::statOn = (std::stoi(envStat) == 1);
        }
    }

    void LogGlobalEnv()
    {
        LOG_DEBUG("Environment variables are: [{}: {}], [{}: {}], [{}: {}], [{}: {}], [{}: {}], [{}: {}], "
                  "[{}: {}], [{}: {}], [{}: {}], [{}: {}], [{}: {}], [{}: {}], [{}: {}]",
                  RecEnvNames::APPLY_GRADIENTS_STRATEGY, GlobalEnv::applyGradientsStrategy,
                  RecEnvNames::ACL_TIMEOUT, GlobalEnv::aclTimeout,
                  RecEnvNames::HD_CHANNEL_SIZE, GlobalEnv::hdChannelSize,
                  RecEnvNames::FIND_OFFSET_V2, GlobalEnv::findOffsetV2,
                  RecEnvNames::FIND_OFFSET_V3, GlobalEnv::findOffsetV3,
                  RecEnvNames::KEY_PROCESS_THREAD_NUM, GlobalEnv::keyProcessThreadNum,
                  RecEnvNames::MAX_UNIQUE_THREAD_NUM, GlobalEnv::maxUniqueThreadNum,
                  RecEnvNames::FAST_UNIQUE, GlobalEnv::fastUnique,
                  RecEnvNames::UPDATE_EMB_V2, GlobalEnv::updateEmbV2,
                  RecEnvNames::HOT_EMB_UPDATE_STEP, GlobalEnv::hotEmbUpdateStep,
                  RecEnvNames::GLOG_STDERR_THRESHOLD, GlobalEnv::glogStderrthreshold,
                  RecEnvNames::USE_COMBINE_FAAE, GlobalEnv::useCombineFaae,
                  RecEnvNames::STAT_ON, GlobalEnv::statOn);
    }
}