/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <sstream>
#include <dsmi_common_interface.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <driver/ascend_hal_define.h>

#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace
{
    int32_t GetLogicID(uint32_t phyid)
    {
        uint32_t logicId;
        int32_t ret = dsmi_get_logicid_from_phyid(phyid, &logicId);
        if (ret != 0) {
            return ret;
        }
        return logicId;
    }

    uint32_t GetDeviceCount()
    {
        uint32_t count;
        aclError ec = aclrtGetDeviceCount(&count);
        if (ec != 0) {
            throw std::runtime_error("The failed to get device count.");
        }
        return count;
    }

    std::string GetChipName(uint32_t devID)
    {
        int ret = 0;
        struct dsmi_chip_info_stru info = {{ 0 },
                                           { 0 },
                                           { 0 }};
        ret = dsmi_get_chip_info(devID, &info);
        if (ret == 0) {
            std::stringstream ss;
            ss << info.chip_name;
            return ss.str();
        }

        throw std::runtime_error("dsmi_get_chip_info failed, ret = " + std::to_string(ret));
    }

    PYBIND11_MODULE(common_binding, m)
    {
        m.def("get_logic_id", &GetLogicID, py::arg("physic_id"));
        m.def("get_device_count", &GetDeviceCount);
        m.def("get_chip_name", &GetChipName, py::arg("device_id"));
    }
}