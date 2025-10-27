/* Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

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

#include <sstream>
#include <dsmi_common_interface.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include "log/logger.h"
#include "common_func.h"

namespace MxRec {
    const int GLOG_MAX_BUF_SIZE = 1024;
    const char* HUGE_TLB_ENABLE = "HUGE_TLB_ENABLE";

    uint32_t GetDeviceCount()
    {
        uint32_t count;
        aclError ec = aclrtGetDeviceCount(&count);
        if (ec != 0) {
            throw std::runtime_error("Failed to get device count, error code: " + std::to_string(ec));
        }
        if (count == 0) {
            throw std::runtime_error("No available devices found");
        }
        return count;
    }

    std::string GetChipName(uint32_t devID)
    {
        if (devID > (GetDeviceCount() - 1)) {
            throw std::runtime_error("Failed to get chip name for device ID: " + std::to_string(devID));
        }
        int ret = 0;
        struct dsmi_chip_info_stru info = {{ 0 },
                                           { 0 },
                                           { 0 }};
        ret = dsmi_get_chip_info(devID, &info);
        if (ret == 0) {
            std::stringstream ss;
            ss << info.chip_name;
            LOG_DEBUG("dsmi_get_chip_info successful, ret = {}, chip_name = {}", ret, ss.str());
            return ss.str();
        }

        throw std::runtime_error("dsmi_get_chip_info failed, ret = " + std::to_string(ret));
    }
}

