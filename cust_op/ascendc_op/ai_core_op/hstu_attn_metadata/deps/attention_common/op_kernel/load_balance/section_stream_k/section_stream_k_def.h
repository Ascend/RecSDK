/* Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.
============================================================================== */

/*!
 * \file section_stream_k_def.h
 * \brief SectionStreamK结构体定义
 */

#ifndef SECTION_STREAM_K_DEF_H
#define SECTION_STREAM_K_DEF_H

#include <vector>
#include "../base_info.h"
#include "../load_balance_common.h"

namespace load_balance {

/******************************** RETURN CODE ********************************/
#define SECTION_STREAM_K_SUCCESS 0
#define SECTION_STREAM_K_ERROR_INVALID_PARAM 1
#define SECTION_STREAM_K_ERROR_AIV_LESS_THAN_AIC 2
/****************************************************************************/

struct SectionStreamKParam : GeneralBalanceParam {
    // 可用于当前任务分段估算的 L2 容量（Byte）；0 表示不按 L2 划分 section。
    uint32_t l2Byte{128 * 1024U * 1024U};
    // 每个有效 S1G 行的一次性固定开销；一行即使含多个 S2 块也只计一次。
    int64_t v0Cost{0};
};

// 一个 section 内 FA（主注意力计算）在 AIC 上的分核边界。
struct SectionStreamKFaResult {
    uint32_t usedCoreNum{0U};                         // FA中使用的AIC数量
    std::vector<uint32_t> bN2End{};                   // 每个核处理数据的BN2结束点
    std::vector<uint32_t> gS1End{};                   // 每个核处理数据的GS1结束点
    std::vector<uint32_t> s2End{};                    // 每个核处理数据的S2结束点
    std::vector<uint32_t> firstFdDataWorkspaceIdx{};  // 每个AIC产生的首个FD中间结果在workspace中的序号

    explicit SectionStreamKFaResult(uint32_t aicNum)
        : bN2End(aicNum),
          gS1End(aicNum),
          s2End(aicNum),
          firstFdDataWorkspaceIdx(aicNum)
    {
    }
};

// 一个 section 内 FD（跨 AIC 的 S2 分片归约）任务及其 AIV 分核信息。
struct SectionStreamKFdResult {
    uint32_t usedVecNum{0U};  // 归约过程中使用的AIV数量
    // 1、归约任务的索引信息
    std::vector<uint32_t> bN2Idx{};  // 每个归约任务的BN2索引，脚标为归约任务的序号，最大为核数-1
    std::vector<uint32_t> gS1Idx{};        // 每个归约任务的GS1索引，脚标为归约任务的序号
    std::vector<uint32_t> workspaceIdx{};  // 每个归约任务在workspace中的存放位置
    std::vector<uint32_t> s2SplitNum{};    // 每个归约任务的S2核间切分份数，脚标为归约任务的序号
    // 2、FD kernel阶段，归约任务的分核信息
    std::vector<uint32_t> taskIdx{};  // 每个AIV处理的归约任务的对应ID
    std::vector<uint32_t> mStart{};   // 每个AIV处理的归约任务的M轴相对起点
    std::vector<uint32_t> mLen{};     // 每个AIV处理的归约任务的M轴行数

    explicit SectionStreamKFdResult(uint32_t aicNum, uint32_t aivNum)
        : bN2Idx(aicNum),
          gS1Idx(aicNum),
          workspaceIdx(aicNum),
          s2SplitNum(aicNum),
          taskIdx(aivNum),
          mStart(aivNum),
          mLen(aivNum)
    {
    }
};

struct SectionStreamKResult {
    uint32_t sectionNum{0U};                                // 按L2容量划分的section数量
    std::vector<SectionStreamKFaResult> sectionFaResult{};  // 每个section一份FA分核结果
    std::vector<SectionStreamKFdResult> sectionFdResult{};  // 与FA结果一一对应的FD归约结果
};

}  // namespace load_balance

#endif
