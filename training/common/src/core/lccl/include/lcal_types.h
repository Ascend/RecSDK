/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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
#ifndef LCAL_TYPES_H
#define LCAL_TYPES_H

#include <map>
#include <string>
#include <set>


namespace Lcal {
constexpr int LCAL_SUCCESS = 0;
constexpr int LCAL_ERROR_NOT_INITIALIZED = -1;
constexpr int LCAL_ERROR_ASDRT = -2;
constexpr int LCAL_ERROR_PARA_CHECK_FAIL = -3;
constexpr int LCAL_ERROR_INTERNAL = -4;
constexpr int LCAL_ERROR_TIMEOUT = -5;
constexpr int LCAL_ERROR_NOT_FOUND = -7;
constexpr int64_t LCAL_INVALID_VALUE = -1;

constexpr int LCAL_BUFF_BYTES = 404 * 1024 * 1024;  // 4MB to store meta info, 400MB to store data

enum class ChipName {
    CHIP_310P3 = 0,
    CHIP_910B1,
    CHIP_910B2,
    CHIP_910B3,
    CHIP_910B4,
    CHIP_910B41,
    CHIP_910B2C,
    CHIP_910_9391,
    CHIP_910_9381,
    CHIP_910_9392,
    CHIP_910_9382,
    CHIP_910_9372,
    CHIP_910_9361,
    RESERVED,
};

const std::set<ChipName> regularChip = {
    ChipName::CHIP_310P3,
    ChipName::CHIP_910B1,
    ChipName::CHIP_910B2,
    ChipName::CHIP_910B3,
    ChipName::CHIP_910B4,
    ChipName::CHIP_910B41,
    ChipName::CHIP_910B2C
};

enum class PhysicalLink {
    HCCS = 0,
    PCIE = 1,
    RESERVED,
};

// 包含 物理链路、芯片名称 信息。
struct PhysicalInfo {
    ChipName chipName = ChipName::RESERVED;
    PhysicalLink physicalLink = PhysicalLink::RESERVED;
    uint32_t coreNum = 0;
};

} // namespace Lcal
#endif // LCAL_TYPES_H
