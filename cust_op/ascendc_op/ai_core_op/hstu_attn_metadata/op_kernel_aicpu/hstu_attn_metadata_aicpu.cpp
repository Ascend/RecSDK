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
 * \file hstu_attn_metadata_aicpu.cpp
 * \brief
 */

#include "log.h"
#include "status.h"
#include <algorithm>
#include <numeric>
#include <cstdio>
#include <cmath>
#include "hstu_attn_metadata.h"
#include "hstu_attn_metadata_aicpu.h"
#include "aicpu_common.h"
#include "fa_adjust_sinner_souter.h"

#define FA_KERNEL_STATUS_OK 0
#define FA_KERNEL_STATUS_PARAM_INVALID 1

using namespace optiling;

namespace aicpu {
uint32_t HstuAttnMetadataCpuKernel::Compute(CpuKernelContext& ctx)
{
    KERNEL_LOG_INFO("Do C Test");
    bool success = Prepare(ctx);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Prepare data failed!");

    load_balance::SectionStreamKResult splitRes{};
    success = BalanceSchedule(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Schedule load balance failed!");

    success = GenMetadata(splitRes);
    KERNEL_CHECK_FALSE(success, FA_KERNEL_STATUS_PARAM_INVALID, "Generate balance result failed!");

    return FA_KERNEL_STATUS_OK;
}

bool HstuAttnMetadataCpuKernel::Prepare(CpuKernelContext& ctx)
{
    // input
    cuSeqlensQ_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensQ));
    cuSeqlensKv_ = ctx.Input(static_cast<uint32_t>(ParamId::cuSeqlensKv));
    sequsedQ_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedQ));
    sequsedKv_ = ctx.Input(static_cast<uint32_t>(ParamId::sequsedKv));
    // output
    metadata_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

    KERNEL_CHECK_FALSE((metadata_ != nullptr && metadata_->GetData() != nullptr), false, "metadata is empty");

    bool requiredAttrs =
        GetAttrValue(ctx, "num_heads_q", numHeadsQ_) && GetAttrValue(ctx, "num_heads_kv", numHeadsKv_) &&
        GetAttrValue(ctx, "head_dim", headDim_) && GetAttrValue(ctx, "soc_version", socVersion_) &&
        GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
    KERNEL_CHECK_FALSE(requiredAttrs, false, "Missing Required attrs missing!");

    // HSTU 仅支持 MHA (G=1)，不支持 GQA 分核下沉。
    KERNEL_CHECK_FALSE(numHeadsQ_ == numHeadsKv_, false,
                       "HSTU mode requires MHA: num_heads_q must equal num_heads_kv (G=1)");

    // attributes optional
    GetAttrValueOpt(ctx, "batch_size", batchSize_);
    GetAttrValueOpt(ctx, "max_seqlen_q", maxSeqlenQ_);
    GetAttrValueOpt(ctx, "max_seqlen_kv", maxSeqlenKv_);
    GetAttrValueOpt(ctx, "mask_mode", maskMode_);
    GetAttrValueOpt(ctx, "win_left", winLeft_);
    GetAttrValueOpt(ctx, "win_right", winRight_);
    GetAttrValueOpt(ctx, "layout_q", layoutQ_);
    GetAttrValueOpt(ctx, "layout_kv", layoutKv_);
    GetAttrValueOpt(ctx, "layout_out", layoutOut_);
    return ParamsInit();
}

bool HstuAttnMetadataCpuKernel::ParamsInit()
{
    InitDeviceInfo();
    InitBaseInfo();
    InitLoadBalanceParams();
    return true;
}

void HstuAttnMetadataCpuKernel::InitDeviceInfo()
{
    deviceInfo.aicCoreMaxNum = aicCoreNum_;
    deviceInfo.aivCoreMaxNum = aivCoreNum_;
    deviceInfo.aicCoreMinNum = aicCoreNum_;
    deviceInfo.aivCoreMinNum = aivCoreNum_;
}

void HstuAttnMetadataCpuKernel::InitLoadBalanceParams()
{
    // 仅 HSTU 模式：分核基本块锁定为 HSTU 反向编译期块大小 (Rk/Cq)，不走 AdjustSinnerAndSouter、不乘 aiv/aic。
    // HSTU host (hstu_backward_v2.cpp::TilingKeySet) 依据 max(dimQK, dimGV) 选 TILE_K：
    //   > 128 → TILE_K=256 → L1TileShape<128,64,256> → Rk(seqK 行块)=get<1>=64, Cq(seqQ 列块)=get<0>=128
    //   否则   → TILE_K=128 → L1TileShape<256,128,128> → Rk=get<1>=128, Cq=get<0>=256
    // 约定：调用方传入的 head_dim 必须是 max(dimQK, dimGV)，才能与 HSTU 编译期块一致。
    // mBaseSize (= Rk) 是硬约束：HSTU kernel 的 MetadataRowBlockScheduler 会校验 HEAD[mBaseSize]==编译期行块，
    // 不一致则该核不出任务。s2BaseSize (= Cq) 是软约束，仅影响 no-FD 下的负载代价。
    if (headDim_ > 128) {    // 128: HSTU TILE_K 阈值
        mBaseSize_ = 64U;    // Rk
        s2BaseSize_ = 128U;  // Cq
    } else {
        mBaseSize_ = 128U;   // Rk
        s2BaseSize_ = 256U;  // Cq
    }
    param.mBaseSize = mBaseSize_;
    param.s2BaseSize = s2BaseSize_;
    param.l2Byte = 96U * 1024U * 1024U;  // 96: 96MB, 1024: Mb2Kb, 1024:Kb2Mb —— 保留多 section
    param.fdOn = false;                  // HSTU 不支持 FD / stream-K，FA 记录必落整行边界
}
void HstuAttnMetadataCpuKernel::InitBaseInfo()
{
    baseInfo.batchSize = batchSize_;
    baseInfo.querySeqSize = maxSeqlenQ_;
    baseInfo.queryHeadNum = numHeadsQ_;
    baseInfo.kvSeqSize = maxSeqlenKv_;
    baseInfo.kvHeadNum = numHeadsKv_;
    baseInfo.headDim = headDim_;
    baseInfo.attenMaskFlag = (maskMode_ != 0);
    baseInfo.sparseMode = static_cast<uint32_t>(maskMode_);
    baseInfo.preToken = (winLeft_ == -1) ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(winLeft_);
    baseInfo.nextToken = (winRight_ == -1) ? std::numeric_limits<uint32_t>::max() : static_cast<uint32_t>(winRight_);
    baseInfo.layoutQuery = load_balance::ConvertToLayout(layoutQ_);
    baseInfo.layoutKv = load_balance::ConvertToLayout(layoutKv_);
    baseInfo.queryType = load_balance::DataType::FP16;
    baseInfo.kvType = load_balance::DataType::FP16;
    LoadActualQuerySeq();
    LoadActualKvSeq();
}

void HstuAttnMetadataCpuKernel::LoadActualQuerySeq()
{
    baseInfo.actualQuerySeqSize.clear();
    baseInfo.isCumulativeQuerySeq = (layoutQ_ == "TND" || layoutQ_ == "NTD");

    if (sequsedQ_ != nullptr && sequsedQ_->GetData() != nullptr) {
        batchSize_ = sequsedQ_->GetTensorShape()->GetDimSize(0);
        baseInfo.batchSize = batchSize_;
        auto tmpSeq = GetTensorDataAsInt64(sequsedQ_, sequsedQ_->GetTensorShape()->GetDimSize(0));
        baseInfo.querySeqSize = static_cast<uint32_t>(*std::max_element(tmpSeq.begin(), tmpSeq.end()));
        baseInfo.actualQuerySeqSize.assign(tmpSeq.begin(), tmpSeq.end());
        if (baseInfo.isCumulativeQuerySeq) {
            std::partial_sum(tmpSeq.begin(), tmpSeq.end(), baseInfo.actualQuerySeqSize.begin());
        }
    } else if (cuSeqlensQ_ != nullptr && cuSeqlensQ_->GetData() != nullptr) {
        batchSize_ = cuSeqlensQ_->GetTensorShape()->GetDimSize(0) - 1U;
        baseInfo.batchSize = batchSize_;
        auto tmpSeq = GetTensorDataAsInt64(cuSeqlensQ_, cuSeqlensQ_->GetTensorShape()->GetDimSize(0));
        baseInfo.actualQuerySeqSize.assign(tmpSeq.begin() + 1, tmpSeq.end());
        baseInfo.querySeqSize = 0U;
        for (size_t i = 1; i < tmpSeq.size(); ++i) {
            auto seq = (baseInfo.isCumulativeQuerySeq) ? tmpSeq[i] - tmpSeq[i - 1] : tmpSeq[i];
            baseInfo.querySeqSize = std::max(baseInfo.querySeqSize, static_cast<uint32_t>(seq));
        }
    }
    return;
}

void HstuAttnMetadataCpuKernel::LoadActualKvSeq()
{
    baseInfo.actualKvSeqSize.clear();
    baseInfo.isCumulativeKvSeq = (layoutKv_ == "TND" || layoutKv_ == "NTD");

    if (sequsedKv_ != nullptr && sequsedKv_->GetData() != nullptr) {
        auto tmpSeq = GetTensorDataAsInt64(sequsedKv_, sequsedKv_->GetTensorShape()->GetDimSize(0));
        baseInfo.kvSeqSize = static_cast<uint32_t>(*std::max_element(tmpSeq.begin(), tmpSeq.end()));
        baseInfo.actualKvSeqSize.assign(tmpSeq.begin(), tmpSeq.end());
        if (baseInfo.isCumulativeKvSeq) {
            std::partial_sum(tmpSeq.begin(), tmpSeq.end(), baseInfo.actualKvSeqSize.begin());
        }
    } else if (cuSeqlensKv_ != nullptr && cuSeqlensKv_->GetData() != nullptr) {
        auto tmpSeq = GetTensorDataAsInt64(cuSeqlensKv_, cuSeqlensKv_->GetTensorShape()->GetDimSize(0));
        baseInfo.actualKvSeqSize.assign(tmpSeq.begin() + 1, tmpSeq.end());
        baseInfo.kvSeqSize = 0U;
        for (size_t i = 1; i < tmpSeq.size(); ++i) {
            auto seq = (baseInfo.isCumulativeKvSeq) ? tmpSeq[i] - tmpSeq[i - 1] : tmpSeq[i];
            baseInfo.kvSeqSize = std::max(baseInfo.kvSeqSize, static_cast<uint32_t>(seq));
        }
    }
    return;
}

bool HstuAttnMetadataCpuKernel::BalanceSchedule(load_balance::SectionStreamKResult& splitRes)
{
    return load_balance::SectionStreamK::Compute(deviceInfo, baseInfo, param, splitRes) == SECTION_STREAM_K_SUCCESS;
}

bool HstuAttnMetadataCpuKernel::GenMetadata(load_balance::SectionStreamKResult& splitRes)
{
    detail::FaMetadata faMetadata(metadata_->GetData(), splitRes.sectionNum);
    faMetadata.Clear();  // set to all 0

    faMetadata.SetHeadMetadata(HEAD_SECTION_NUM_INDEX, splitRes.sectionNum);
    faMetadata.SetHeadMetadata(HEAD_M_BASE_SIZE_INDEX, mBaseSize_);
    faMetadata.SetHeadMetadata(HEAD_S2_BASE_SIZE_INDEX, s2BaseSize_);
    if (std::any_of(splitRes.sectionFdResult.begin(), splitRes.sectionFdResult.end(),
                    [](load_balance::SectionStreamKFdResult result) { return result.usedVecNum > 0U; })) {
        faMetadata.SetHeadMetadata(HEAD_IS_FD_INDEX, 1U);
    }

    load_balance::SectionStreamKFaResult dummyHead{static_cast<uint32_t>(aicCoreNum_)};  // all zeror dummy head
    for (uint32_t secIdx = 0; secIdx < splitRes.sectionNum; ++secIdx) {
        auto& faRes = splitRes.sectionFaResult[secIdx];
        for (uint32_t aicIdx = 0; aicIdx < faRes.usedCoreNum; ++aicIdx) {
            auto& prevFaRes = (secIdx == 0U) ? dummyHead : splitRes.sectionFaResult[secIdx - 1U];
            auto prevLastCore = (secIdx == 0U) ? 0U : prevFaRes.usedCoreNum - 1U;
            FA_METADATA_T bn2Start = (aicIdx == 0) ? prevFaRes.bN2End[prevLastCore] : faRes.bN2End[aicIdx - 1U];
            FA_METADATA_T mStart = (aicIdx == 0) ? prevFaRes.gS1End[prevLastCore] : faRes.gS1End[aicIdx - 1U];
            FA_METADATA_T s2Start = (aicIdx == 0) ? prevFaRes.s2End[prevLastCore] : faRes.s2End[aicIdx - 1U];

            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_START_INDEX, bn2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_START_INDEX, mStart);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_START_INDEX, s2Start);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_BN2_END_INDEX, faRes.bN2End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_M_END_INDEX, faRes.gS1End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_S2_END_INDEX, faRes.s2End[aicIdx]);
            faMetadata.SetFaMetadata(secIdx, aicIdx, FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX,
                                     faRes.firstFdDataWorkspaceIdx[aicIdx]);
        }

        auto& fdRes = splitRes.sectionFdResult[secIdx];
        for (uint32_t aivIdx = 0; aivIdx < fdRes.usedVecNum; ++aivIdx) {
            uint32_t t = fdRes.taskIdx[aivIdx];
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_BN2_IDX_INDEX, fdRes.bN2Idx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_IDX_INDEX, fdRes.gS1Idx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_IDX_INDEX, fdRes.workspaceIdx[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_WORKSPACE_NUM_INDEX, fdRes.s2SplitNum[t]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_START_INDEX, fdRes.mStart[aivIdx]);
            faMetadata.SetFdMetadata(secIdx, aivIdx, FD_M_NUM_INDEX, fdRes.mLen[aivIdx]);
        }
    }
    return true;
}

namespace {
static const char* kernelType = "HstuAttnMetadata";
REGISTER_CPU_KERNEL(kernelType, HstuAttnMetadataCpuKernel);
}  // namespace

}  // namespace aicpu
