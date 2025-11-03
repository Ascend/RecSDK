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

#include <dsmi_common_interface.h>
#include <driver/ascend_hal_define.h>
#include <limits>

#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "log/logger.h"
#include "common_func/common_func.h"

namespace py = pybind11;
namespace {

    int32_t GetLogicID(uint32_t phyid)
    {
        uint32_t logicId;
        int32_t ret = dsmi_get_logicid_from_phyid(phyid, &logicId);
        if (ret != 0) {
            LOG_ERROR("dsmi_get_logicid_from_phyid get logicId failed, ret:{}", ret);
            return -1;
        }
        if (logicId > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            LOG_ERROR("dsmi_get_logicid_from_phyid logicId invalid, logicId:{}", logicId);
            return -1;
        }
        return static_cast<int32_t>(logicId);
    }

    PYBIND11_MODULE(common_binding, m)
    {
        m.def("get_logic_id", &GetLogicID, py::arg("physic_id"));
        m.def("get_device_count", &MxRec::GetDeviceCount);
        m.def("get_chip_name", &MxRec::GetChipName, py::arg("device_id"));
    }
}