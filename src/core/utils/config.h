/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: config module
 * Author: MindX SDK
 * Create: 2023
 * History: NA
 */

#ifndef MXREC_CONFIG_H
#define MXREC_CONFIG_H

#include <string>

namespace MxRec {
    namespace RecEnvNames {
        const char *const APPLY_GRADIENTS_STRATEGY = "APPLY_GRADIENTS_STRATEGY";
        const char *const ACL_TIMEOUT = "AclTimeout";
        const char *const HD_CHANNEL_SIZE = "HD_CHANNEL_SIZE";
        const char *const FIND_OFFSET_V2 = "FIND_OFFSET_V2";
        const char *const FIND_OFFSET_V3 = "FIND_OFFSET_V3";
        const char *const KEY_PROCESS_THREAD_NUM = "KEY_PROCESS_THREAD_NUM";
        const char *const MAX_UNIQUE_THREAD_NUM = "MAX_UNIQUE_THREAD_NUM";
        const char *const FAST_UNIQUE = "FAST_UNIQUE";
        const char *const UPDATE_EMB_V2 = "UpdateEmb_V2";
        const char *const HOT_EMB_UPDATE_STEP = "HOT_EMB_UPDATE_STEP";
        const char *const GLOG_STDERR_THRESHOLD = "GLOG_stderrthreshold";
        const char *const USE_COMBINE_FAAE = "USE_COMBINE_FAAE";
        const char *const STAT_ON = "STAT_ON";
    };

    namespace ApplyGradientsStrategyOptions {
        extern const std::string DIRECT_APPLY;
        extern const std::string SUM_SAME_ID_GRADIENTS_AND_APPLY;
    };

    struct GlobalEnv {
        static std::string applyGradientsStrategy;
        static int aclTimeout;
        static int hdChannelSize;
        static bool findOffsetV2;
        static bool findOffsetV3;
        static int keyProcessThreadNum;
        static int maxUniqueThreadNum;
        static bool fastUnique;
        static bool updateEmbV2;
        static int hotEmbUpdateStep;
        static int glogStderrthreshold;
        static bool useCombineFaae;
        static bool statOn;
    };

    void ConfigGlobalEnv();
    void LogGlobalEnv();
}

#endif

