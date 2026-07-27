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
 * \file hstu_attn_metadata.h
 * \brief
 */

#ifndef hstu_attn_METADATA_H
#define hstu_attn_METADATA_H

#include <cstdint>
#include <cassert>

namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_NUM = 36U;
constexpr uint32_t AIV_CORE_NUM = 72U;
constexpr uint32_t FA_META_SIZE = 1024U;
using FA_METADATA_T = uint32_t;

constexpr uint32_t HEAD_METADATA_STRIDE = 16U;
constexpr uint32_t FA_METADATA_STRIDE = 16U;
constexpr uint32_t FD_METADATA_STRIDE = 16U;

// Head Metadata Index Definitions
constexpr uint32_t HEAD_SECTION_NUM_INDEX = 0U;
constexpr uint32_t HEAD_IS_FD_INDEX = 1U;
constexpr uint32_t HEAD_M_BASE_SIZE_INDEX = 2U;
constexpr uint32_t HEAD_S2_BASE_SIZE_INDEX = 3U;

// FA Metadata Index Definitions
constexpr uint32_t FA_BN2_START_INDEX = 0U;
constexpr uint32_t FA_M_START_INDEX = 1U;
constexpr uint32_t FA_S2_START_INDEX = 2U;
constexpr uint32_t FA_BN2_END_INDEX = 3U;
constexpr uint32_t FA_M_END_INDEX = 4U;
constexpr uint32_t FA_S2_END_INDEX = 5U;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6U;

// FD Metadata Index Definitions
constexpr uint32_t FD_BN2_IDX_INDEX = 0U;
constexpr uint32_t FD_M_IDX_INDEX = 1U;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 2U;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 3U;
constexpr uint32_t FD_M_START_INDEX = 4U;
constexpr uint32_t FD_M_NUM_INDEX = 5U;

namespace detail {
struct FaMetadata {
    uint32_t sectionNum;
    FA_METADATA_T* headMetadata;  // [HEAD_METADATA_STRIDE];
    FA_METADATA_T* faMetadata;    // [sectionNum][AIC_CORE_NUM][FA_METADATA_STRIDE];
    FA_METADATA_T* fdMetadata;    // [sectionNum][AIV_CORE_NUM][FD_METADATA_STRIDE];
    FaMetadata(void* metadataPtr, uint32_t sectionNum)
        : sectionNum(sectionNum),
          headMetadata(static_cast<FA_METADATA_T*>(metadataPtr)),
          faMetadata(headMetadata + HEAD_METADATA_STRIDE),
          fdMetadata(faMetadata + sectionNum * AIC_CORE_NUM * FA_METADATA_STRIDE)
    {
        headMetadata[0] = sectionNum;
    }

    void Clear()
    {
        for (size_t i = 0; i < HEAD_METADATA_STRIDE; ++i) {
            headMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * AIC_CORE_NUM * FA_METADATA_STRIDE; ++i) {
            faMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * AIV_CORE_NUM * FD_METADATA_STRIDE; ++i) {
            fdMetadata[i] = 0U;
        }
    }

    void SetHeadMetadata(uint32_t metaIdx, uint32_t val)
    {
        assert(metaIdx < HEAD_METADATA_STRIDE);
        headMetadata[metaIdx] = val;
    }

    uint32_t GetHeadMetadata(uint32_t metaIdx)
    {
        assert(metaIdx < HEAD_METADATA_STRIDE);
        return headMetadata[metaIdx];
    }

    void SetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_STRIDE);
        faMetadata[sectionIdx * AIC_CORE_NUM * FA_METADATA_STRIDE + aicIdx * FA_METADATA_STRIDE + metaIdx] = val;
    }

    uint32_t GetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_STRIDE);
        return faMetadata[AIC_CORE_NUM * FA_METADATA_STRIDE * sectionIdx + FA_METADATA_STRIDE * aicIdx + metaIdx];
    }

    void SetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < FD_METADATA_STRIDE);
        fdMetadata[AIV_CORE_NUM * FD_METADATA_STRIDE * sectionIdx + FD_METADATA_STRIDE * aivIdx + metaIdx] = val;
    }

    uint32_t GetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < FD_METADATA_STRIDE);
        return fdMetadata[AIV_CORE_NUM * FD_METADATA_STRIDE * sectionIdx + FD_METADATA_STRIDE * aivIdx + metaIdx];
    }
};
}  // namespace detail

}  // namespace optiling

#endif
