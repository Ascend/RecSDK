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

#ifndef MXREC_CONFIG_H
#define MXREC_CONFIG_H

namespace MxRec {
    namespace RecEnvNames {
        const char *const ACL_TIMEOUT = "AclTimeout";
        const char *const HD_CHANNEL_SIZE = "HD_CHANNEL_SIZE";
        const char *const KEY_PROCESS_THREAD_NUM = "KEY_PROCESS_THREAD_NUM";
        const char *const MAX_UNIQUE_THREAD_NUM = "MAX_UNIQUE_THREAD_NUM";
        const char *const FAST_UNIQUE = "FAST_UNIQUE";
        const char *const HOT_EMB_UPDATE_STEP = "HOT_EMB_UPDATE_STEP";
        const char *const GLOG_STDERR_THRESHOLD = "GLOG_stderrthreshold";
        const char *const USE_COMBINE_FAAE = "USE_COMBINE_FAAE";
        const char *const RECORD_KEY_COUNT = "RECORD_KEY_COUNT";
        const char *const USE_SHM_SWAP = "USE_SHM_SWAP";
        const char *const HUGE_TLB_ENABLE = "HUGE_TLB_ENABLE";
        const char *const SSD_SAVE_COMPACT_LEVEL = "SSD_SAVE_COMPACT_LEVEL";
    };

    struct GlobalEnv {
        static int aclTimeout;
        static int hdChannelSize;
        static int keyProcessThreadNum;
        static int maxUniqueThreadNum;
        static bool fastUnique;
        static int hotEmbUpdateStep;
        static int glogStderrthreshold;
        static bool useCombineFaae;
        static bool recordKeyCount;
        static bool useShmSwap;
        static bool hugeTlbEnable;
        static int ssdSaveCompactLevel;
    };

    void ConfigGlobalEnv();
    void LogGlobalEnv();
}

#endif

