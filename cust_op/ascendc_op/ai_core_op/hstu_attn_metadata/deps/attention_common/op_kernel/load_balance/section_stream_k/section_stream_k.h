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
 * \file section_stream_k.h
 * \brief SectionStreamK对外接口
 */

#ifndef SECTION_STREAM_K_H
#define SECTION_STREAM_K_H

#include <vector>
#include "../base_info.h"
#include "section_stream_k_def.h"
#include "section_stream_k_impl.h"

namespace load_balance {

class SectionStreamK {
public:
    // 计算分核结果。deviceInfo 描述可用 AIC/AIV，baseInfo 描述形状和 mask，param 描述基本块及策略参数。
    inline static uint32_t Compute(const DeviceInfo& deviceInfo, const IBaseInfo& baseInfo,
                                   const SectionStreamKParam& param, SectionStreamKResult& result);
};

inline uint32_t SectionStreamK::Compute(const DeviceInfo& deviceInfo, const IBaseInfo& baseInfo,
                                        const SectionStreamKParam& param, SectionStreamKResult& result)
{
    SectionStreamKImpl impl{};
    // 参数校验并设置默认开销模型。
    auto ret = impl.SetParam(param);
    if (ret != SECTION_STREAM_K_SUCCESS) {
        return ret;
    }

    if (deviceInfo.aivCoreMaxNum < deviceInfo.aicCoreMaxNum) {
        return SECTION_STREAM_K_ERROR_AIV_LESS_THAN_AIC;
    }

    // 内部结果同时包含调度搜索所需的统计信息；对外仅保留 kernel 消费的 FA/FD 元数据。
    auto implResult = impl.Compute(deviceInfo, baseInfo);

    result.sectionNum = implResult.size();
    for (size_t i = 0; i < result.sectionNum; ++i) {
        SectionStreamKFaResult tmpFa(deviceInfo.aicCoreMaxNum);
        tmpFa.usedCoreNum = implResult[i].usedCoreNum;
        tmpFa.bN2End = implResult[i].bN2End;
        tmpFa.gS1End = implResult[i].gS1End;
        tmpFa.s2End = implResult[i].s2End;
        tmpFa.firstFdDataWorkspaceIdx = implResult[i].firstFdDataWorkspaceIdx;
        result.sectionFaResult.emplace_back(tmpFa);

        SectionStreamKFdResult tmpFd(deviceInfo.aicCoreMaxNum, deviceInfo.aivCoreMaxNum);
        tmpFd.usedVecNum = implResult[i].usedVecNum;
        tmpFd.bN2Idx = implResult[i].bN2Idx;
        tmpFd.gS1Idx = implResult[i].mIdx;
        tmpFd.workspaceIdx = implResult[i].workspaceIdx;
        tmpFd.s2SplitNum = implResult[i].s2SplitNum;
        tmpFd.taskIdx = implResult[i].taskIdx;
        tmpFd.mStart = implResult[i].mStart;
        tmpFd.mLen = implResult[i].mLen;
        result.sectionFdResult.emplace_back(tmpFd);
    }

    return SECTION_STREAM_K_SUCCESS;
}

}  // namespace load_balance

#endif
