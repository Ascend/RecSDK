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
 * \file test_aclnn_hstu_attn_metadata_backward.cpp
 * \brief 上板冒烟用例：下发算子并**校验 metadata 内容**。
 *
 * 判定标准（三条全过才算 PASS）：
 *   1. 算子能被正常下发与执行（acl 调用不报错、stream 同步成功）；
 *   2. HEAD 段与入参一致：sectionNum >= 1、isFd == 0、mBaseSize/s2BaseSize 与 head_dim 的档位相符；
 *   3. FA 段的核区间恰好覆盖全部 K 行块且互不重叠（覆盖性 —— 本算子唯一的硬性不变量）。
 *      块数少于核数时切出空核是合法的，故这里不要求「每核非空」。
 *
 * 三个用例：
 *   用例 1 mask_mode=0（no mask）—— 基线；
 *   用例 2 mask_mode=2（arbitrary）—— 成对下发 full_cnt / mask_cnt，cnt 在有效行块上
 *   交替置 1（full/mask 两条张量都真实参与求和），权重恒 1 == no-mask 的 qBlkCnt；
 *   覆盖性同样必须成立，且 metadata 与用例 1 **逐元素一致**（权重等价 ⇒ 切分等价；
 *   若 kernel 仍按占位 0 权重切，quota==0 的按块均切会切出不同区间，此处立刻对不上）；
 *   用例 3 mask_mode=1（causal）—— 下发非零 num_contexts / num_targets / target_group_size
 *   使 FromCtx 解析（含 clamp）与 GetTargetQBlockEnd 的 history/target 双分支全部真实执行；
 *   本数据集下 qBlkCnt=1 且 deltaQK 足够大（tril 后缀覆盖全部 Q 块），任何 causal 配置的
 *   权重恒 = τ = 1 == no-mask 的 qBlkCnt，故 metadata 仍必须与用例 1 逐元素一致。
 *
 * 校验用的读法（FlattenRowBlock / 展平区间）在下面**独立实现一份**，刻意不复用设备侧代码：
 * 写入侧与读取侧各持一份声明，任何一侧改了 stride / 下标 / 展平公式就会立刻对不上。
 *
 * 运行:
 *   bash run.sh --stage=run
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_hstu_attn_metadata_backward.h"

#include "../op_kernel_aicpu/hstu_attn_metadata_backward.h"

using namespace optiling::backward;

#define CHECK_ACL(expr, msg)                                            \
    do {                                                                \
        auto _ret = (expr);                                             \
        if (_ret != ACL_SUCCESS) {                                      \
            printf("[FAIL] %s failed, ret=%d\n", (msg), (int32_t)_ret); \
            return -1;                                                  \
        }                                                               \
    } while (0)

constexpr int32_t kSentinel = 0x5A5A5A5A;

// mask_mode 取值：0 = no mask（用例 1 基线），1 = causal（用例 3，下发 num_contexts / num_targets /
// target_group_size），2 = arbitrary（用例 2，成对下发 full_cnt / mask_cnt）
constexpr int64_t kMaskModeNoMask = 0;
constexpr int64_t kMaskModeCausal = 1;
constexpr int64_t kMaskModeArbitrary = 2;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) {
        size *= dim;
    }
    return size;
}

/*!
 * \brief 期望的块档位（BLOCK_M 沿 K 轴、BLOCK_N 沿 Q 轴）。
 *
 * 独立声明一份真值表，与算子内 DeriveBlockShape 对照：
 *   TILE_K = head_dim > 128 ? 256 : 128
 *   TILE_K=128 → BLOCK_M=128, BLOCK_N=256
 *   TILE_K=256 → BLOCK_M= 64, BLOCK_N=128
 */
struct BlockShape {
    uint32_t blockM;
    uint32_t blockN;
};

BlockShape ExpectedBlockShape(int64_t headDim)
{
    return (headDim > 128) ? BlockShape{64U, 128U} : BlockShape{128U, 256U};
}

/*!
 * \brief 镜像设备侧 MetadataRowBlockScheduler::FlattenRowBlock。
 *
 *     flatten(bn2, m) = Σ_{x<bn2} ceil(seqLenK(x / numHeads) / BLOCK_M) + m
 */
uint32_t FlattenRowBlock(uint32_t bn2, uint32_t m, const std::vector<uint32_t>& rowBlocks, uint32_t numHeads)
{
    uint32_t acc = 0U;
    for (uint32_t x = 0U; x < bn2; ++x) {
        acc += rowBlocks[x / numHeads];
    }
    return acc + m;
}

/*! \brief 按设备侧读法回读并校验 metadata。返回 0 表示通过。 */
int CheckMetadata(const std::vector<int32_t>& meta, int64_t metaElems, const std::vector<int32_t>& cuKHost,
                  uint32_t numHeads, int64_t headDim)
{
    const BlockShape shape = ExpectedBlockShape(headDim);

    // HEAD 指针不依赖 sectionNum，先用 0 建一次视图把 HEAD 读出来。
    MetadataBuffer headBuf = MakeMetadataBuffer(const_cast<int32_t*>(meta.data()), 0U);
    const uint32_t sectionNum = static_cast<uint32_t>(GetHead(headBuf, kHeadSectionNumIdx));
    const int32_t isFd = GetHead(headBuf, kHeadIsFdIdx);
    const int32_t mBaseSize = GetHead(headBuf, kHeadMBaseSizeIdx);
    const int32_t s2BaseSize = GetHead(headBuf, kHeadS2BaseSizeIdx);

    printf("  HEAD[sectionNum]=%u HEAD[isFd]=%d HEAD[mBaseSize]=%d HEAD[s2BaseSize]=%d\n", sectionNum, isFd, mBaseSize,
           s2BaseSize);

    if (sectionNum < 1U) {
        printf("[FAIL] sectionNum=%u，必须 >= 1\n", sectionNum);
        return -1;
    }
    if (isFd != 0) {
        printf("[FAIL] 反向不使用 FD，isFd 必须为 0，实际 %d\n", isFd);
        return -1;
    }
    if (mBaseSize != static_cast<int32_t>(shape.blockM) || s2BaseSize != static_cast<int32_t>(shape.blockN)) {
        printf("[FAIL] 块档位与 head_dim=%ld 不符：期望 mBaseSize=%u s2BaseSize=%u，实际 %d / %d\n",
               static_cast<long>(headDim), shape.blockM, shape.blockN, mBaseSize, s2BaseSize);
        return -1;
    }

    const uint32_t batchSize = static_cast<uint32_t>(cuKHost.size()) - 1U;
    if (sectionNum > batchSize * numHeads) {
        printf("[FAIL] sectionNum=%u 超过上界 batch*numHeads=%u\n", sectionNum, batchSize * numHeads);
        return -1;
    }

    // K 行块数：每个 BN2 的块数只与其 batch 的 seqLenK 有关
    std::vector<uint32_t> rowBlocks(batchSize, 0U);
    for (uint32_t b = 0U; b < batchSize; ++b) {
        const int64_t seqLen = static_cast<int64_t>(cuKHost[b + 1U]) - static_cast<int64_t>(cuKHost[b]);
        rowBlocks[b] = static_cast<uint32_t>((static_cast<uint64_t>(seqLen) + shape.blockM - 1U) / shape.blockM);
    }
    uint32_t totalBlocks = 0U;
    for (uint32_t blocks : rowBlocks) {
        totalBlocks += blocks;
    }
    totalBlocks *= numHeads;

    const uint32_t required = RequiredMetadataElements(sectionNum);
    if (static_cast<int64_t>(required) > metaElems) {
        printf("[FAIL] metadata 缓冲不足：需要 %u，实际 %ld\n", required, static_cast<long>(metaElems));
        return -1;
    }
    MetadataBuffer buf = MakeMetadataBuffer(const_cast<int32_t*>(meta.data()), sectionNum);

    // ---- FD 区必须整段为 0 ----
    for (uint32_t i = 0U; i < static_cast<uint32_t>(sectionNum) * kAivCoreNum * kFdStride; ++i) {
        if (buf.fd[i] != 0) {
            printf("[FAIL] FD 区必须全 0，实际 fd[%u]=%d\n", i, buf.fd[i]);
            return -1;
        }
    }

    // ---- FA 段：逐 section 逐核解析 ----
    struct Interval {
        uint32_t begin;
        uint32_t end;
        uint32_t sec;
        uint32_t core;
    };
    std::vector<Interval> intervals;
    uint32_t usedCores = 0U;

    for (uint32_t sec = 0U; sec < sectionNum; ++sec) {
        for (uint32_t core = 0U; core < kAicCoreNum; ++core) {
            const int32_t s2Start = GetFa(buf, sec, core, kFaS2StartIdx);
            const int32_t s2End = GetFa(buf, sec, core, kFaS2EndIdx);
            const int32_t fdWs = GetFa(buf, sec, core, kFaFirstFdWsIdx);
            if (s2Start != 0 || s2End != 0 || fdWs != 0) {
                printf("[FAIL] section=%u core=%u 的 s2/fdWsIdx 是保留位，必须为 0（实际 %d/%d/%d）\n", sec, core,
                       s2Start, s2End, fdWs);
                return -1;
            }

            bool allZero = true;
            for (uint32_t k = 0U; k < kFaStride; ++k) {
                if (GetFa(buf, sec, core, k) != 0) {
                    allZero = false;
                    break;
                }
            }
            if (!allZero) {
                ++usedCores;
            }

            const uint32_t begin =
                FlattenRowBlock(static_cast<uint32_t>(GetFa(buf, sec, core, kFaBn2StartIdx)),
                                static_cast<uint32_t>(GetFa(buf, sec, core, kFaMStartIdx)), rowBlocks, numHeads);
            const uint32_t end =
                FlattenRowBlock(static_cast<uint32_t>(GetFa(buf, sec, core, kFaBn2EndIdx)),
                                static_cast<uint32_t>(GetFa(buf, sec, core, kFaMEndIdx)), rowBlocks, numHeads);
            if (end < begin) {
                printf("[FAIL] FA 区间倒序：section=%u core=%u 展平为 [%u, %u)\n", sec, core, begin, end);
                return -1;
            }
            if (end > begin) {
                intervals.push_back(Interval{begin, end, sec, core});
            }
        }
    }

    // ---- 覆盖性：有效区间按展平位置排序后必须无缝铺满 [0, totalBlocks) ----
    std::sort(intervals.begin(), intervals.end(),
              [](const Interval& lhs, const Interval& rhs) { return lhs.begin < rhs.begin; });
    uint32_t cursor = 0U;
    for (const Interval& interval : intervals) {
        if (interval.begin != cursor) {
            printf("[FAIL] K 行块覆盖不连续或重叠：期望从 %u 开始，实际区间 [%u, %u) (section=%u core=%u)\n", cursor,
                   interval.begin, interval.end, interval.sec, interval.core);
            return -1;
        }
        cursor = interval.end;
    }
    if (cursor != totalBlocks) {
        printf("[FAIL] K 行块覆盖不完整：期望 %u 块，实际 %u（numHeads=%u）\n", totalBlocks, cursor, numHeads);
        return -1;
    }

    printf("  FA: %u 个有效区间 / %u 个核槽被占用，覆盖全部 %u 个 K 行块\n", static_cast<uint32_t>(intervals.size()),
           usedCores, totalBlocks);
    return 0;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        printf("aclInit failed. ERROR: %d\n", (int32_t)ret);
        return ret;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        printf("aclrtSetDevice failed. ERROR: %d\n", (int32_t)ret);
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("aclrtCreateStream failed. ERROR: %d\n", (int32_t)ret);
        return ret;
    }
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(deviceAddr, static_cast<size_t>(size), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        printf("aclrtMalloc failed. ERROR: %d\n", (int32_t)ret);
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddr, static_cast<size_t>(size), hostData.data(), static_cast<size_t>(size),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        printf("aclrtMemcpy failed. ERROR: %d\n", (int32_t)ret);
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

/*!
 * \brief 两段式下发一次算子并同步回读 metadata（两个用例共用的执行链路）。
 *
 * AICPU 算子自带栈上缓冲，workspaceSize 恒 0，无需申请 workspace。
 * \param metaHost [out] 回读结果（容量 = metaElems）
 * \return 0 表示成功
 */
int RunOpAndReadBack(aclrtStream stream, aclTensor* cuQ, aclTensor* cuK, aclTensor* numCtx, aclTensor* numTgt,
                     int64_t numHeads, int64_t headDim, int64_t maskMode, int64_t targetGroupSize, aclTensor* fullCnt,
                     aclTensor* maskCnt, aclTensor* metaTensor, void* metaDev, int64_t metaElems,
                     std::vector<int32_t>& metaHost)
{
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;

    auto ret = aclnnHstuAttnMetadataBackwardGetWorkspaceSize(cuQ, cuK, numCtx, numTgt, numHeads, headDim, maskMode,
                                                             targetGroupSize, fullCnt, maskCnt, metaTensor,
                                                             &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        printf("[FAIL] aclnnHstuAttnMetadataBackwardGetWorkspaceSize ret=%d\n", (int32_t)ret);
        return -1;
    }
    printf("  workspaceSize=%llu\n", (unsigned long long)workspaceSize);

    ret = aclnnHstuAttnMetadataBackward(nullptr, workspaceSize, executor, stream);
    if (ret != ACL_SUCCESS) {
        printf("[FAIL] aclnnHstuAttnMetadataBackward ret=%d\n", (int32_t)ret);
        return -1;
    }

    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("[FAIL] aclrtSynchronizeStream ret=%d\n", (int32_t)ret);
        return -1;
    }

    ret =
        aclrtMemcpy(metaHost.data(), metaHost.size() * sizeof(metaHost[0]), metaDev,
                    static_cast<size_t>(metaElems * static_cast<int64_t>(sizeof(int32_t))), ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        printf("[FAIL] aclrtMemcpy ret=%d\n", (int32_t)ret);
        return -1;
    }
    return 0;
}

/*! \brief 用例 1：mask_mode=0（no mask）基线；结果写入 metaHost 供用例 2 做等价比对。 */
int RunNoMaskCase(aclTensor* cuQ, aclTensor* cuK, const std::vector<int32_t>& cuKHost, int64_t numHeads,
                  int64_t headDim, int64_t metaElems, const std::vector<int64_t>& metaShape, aclrtStream stream,
                  std::vector<int32_t>& metaHost)
{
    metaHost.assign(static_cast<size_t>(metaElems), kSentinel);
    void* metaDev = nullptr;
    aclTensor* metaTensor = nullptr;
    CHECK_ACL(CreateAclTensor(metaHost, metaShape, &metaDev, aclDataType::ACL_INT32, &metaTensor), "create metadata");

    printf("=== 用例 1：no mask（mask_mode=0）基线 ===\n");
    if (RunOpAndReadBack(stream, cuQ, cuK, nullptr, nullptr, numHeads, headDim, kMaskModeNoMask, 0, nullptr, nullptr,
                         metaTensor, metaDev, metaElems, metaHost) != 0) {
        return -1;
    }
    if (CheckMetadata(metaHost, metaElems, cuKHost, static_cast<uint32_t>(numHeads), headDim) != 0) {
        printf("[FAIL] metadata 内容校验未通过\n");
        return -1;
    }
    printf("[PASS] 算子下发执行成功，metadata(%ld 元素) 内容校验通过\n", (long)metaElems);

    aclDestroyTensor(metaTensor);
    aclrtFree(metaDev);
    return 0;
}

/*!
 * \brief 用例 2：mask_mode=2（arbitrary）—— 成对下发 full_cnt / mask_cnt。
 *
 * cnt 契约：numel = batch * maxBlkCntK（由 cuKHost 与 headDim 推导的块档位自洽计算）。
 * 有效行块上 full/mask 交替置 1（两条张量都真实参与求和），权重恒 1 == no-mask 的
 * qBlkCnt=1（seqLenQ=128 < BLOCK_N=256）；k >= kBlkCnt[b] 的 padding 区恒 0（不会被读取）。
 * 校验：覆盖性 + 与 no-mask 基线逐元素一致（权重等价 ⇒ 切分等价；若 kernel 仍按占位 0
 * 权重切，quota==0 的按块均切会切出不同区间，此处立刻对不上）。
 */
int RunArbitraryCase(aclTensor* cuQ, aclTensor* cuK, const std::vector<int32_t>& cuKHost, int64_t numHeads,
                     int64_t headDim, int64_t metaElems, const std::vector<int64_t>& metaShape, aclrtStream stream,
                     const std::vector<int32_t>& baseline)
{
    // cnt 形状自洽推导：maxBlkCntK = max_b ceil(seqLenK[b] / BLOCK_M)
    const uint32_t blockM = ExpectedBlockShape(headDim).blockM;
    const int64_t batch = static_cast<int64_t>(cuKHost.size()) - 1;
    int64_t maxBlkCntK = 0;
    for (int64_t b = 0; b < batch; ++b) {
        const int64_t seqLenK = static_cast<int64_t>(cuKHost[b + 1]) - cuKHost[b];
        maxBlkCntK = std::max(maxBlkCntK, (seqLenK + blockM - 1) / blockM);
    }

    std::vector<int32_t> fullHost(static_cast<size_t>(batch * maxBlkCntK), 0);
    std::vector<int32_t> maskHost(static_cast<size_t>(batch * maxBlkCntK), 0);
    for (int64_t b = 0; b < batch; ++b) {
        const int64_t seqLenK = static_cast<int64_t>(cuKHost[b + 1]) - cuKHost[b];
        const int64_t kBlkCnt = (seqLenK + blockM - 1) / blockM;
        for (int64_t k = 0; k < kBlkCnt; ++k) {
            if ((k & 1) == 0) {
                fullHost[b * maxBlkCntK + k] = 1;
            } else {
                maskHost[b * maxBlkCntK + k] = 1;
            }
        }
    }
    const std::vector<int64_t> cntShape = {batch * maxBlkCntK};

    std::vector<int32_t> metaHost(static_cast<size_t>(metaElems), kSentinel);
    void* fullDev = nullptr;
    void* maskDev = nullptr;
    void* metaDev = nullptr;
    aclTensor* fullTensor = nullptr;
    aclTensor* maskTensor = nullptr;
    aclTensor* metaTensor = nullptr;
    CHECK_ACL(CreateAclTensor(fullHost, cntShape, &fullDev, aclDataType::ACL_INT32, &fullTensor), "create full_cnt");
    CHECK_ACL(CreateAclTensor(maskHost, cntShape, &maskDev, aclDataType::ACL_INT32, &maskTensor), "create mask_cnt");
    CHECK_ACL(CreateAclTensor(metaHost, metaShape, &metaDev, aclDataType::ACL_INT32, &metaTensor),
              "create metadata(af)");

    printf("=== 用例 2：arbitrary mask（mask_mode=2，cnt 有效行块全 1 权重）===\n");
    if (RunOpAndReadBack(stream, cuQ, cuK, nullptr, nullptr, numHeads, headDim, kMaskModeArbitrary, 0, fullTensor,
                         maskTensor, metaTensor, metaDev, metaElems, metaHost) != 0) {
        return -1;
    }

    // 覆盖性同样必须成立（行块网格不因 mask 改变）
    if (CheckMetadata(metaHost, metaElems, cuKHost, static_cast<uint32_t>(numHeads), headDim) != 0) {
        printf("[FAIL] af: metadata 内容校验未通过\n");
        return -1;
    }
    // 权重等价 ⇒ 切分等价：全 1 权重（== qBlkCnt）应与 no-mask 基线逐元素一致
    if (metaHost != baseline) {
        printf("[FAIL] af 全 1 权重应与 no-mask（qBlkCnt=1）切分逐元素一致，实际存在差异\n");
        return -1;
    }
    printf("[PASS] af 模式跑通：覆盖性校验通过，且与 no-mask 基线逐元素一致\n");

    aclDestroyTensor(fullTensor);
    aclDestroyTensor(maskTensor);
    aclDestroyTensor(metaTensor);
    aclrtFree(fullDev);
    aclrtFree(maskDev);
    aclrtFree(metaDev);
    return 0;
}

/*!
 * \brief 用例 3：mask_mode=1（causal）—— 下发非零 num_contexts / num_targets / target_group_size。
 *
 * 取值设计（batch=4，seqLenQ 恒 128 → qBlkCnt=1，BLOCK_M=128 / BLOCK_N=256）：
 *   numContexts = {64, 0, 128, 32}、numTargets = {64, 0, 128, 64}、targetGroupSize = 64，
 *   使 FromCtx 的非空解析 + clamp、GetTargetQBlockEnd 的 numTarget=0 早退（b1）/ history 段
 *   （b0/b3）/ target 段（b2 的 k=3：kBlockBegin=384 >= historyLen=384）三条路径全部真实执行。
 * 等价性：本数据集 deltaQK（128/128/384/896）足够大，tril 后缀覆盖全部 Q 块（qSeqMin 恒 0），
 *   故每块权重恒 = τ = 1 == no-mask 的 qBlkCnt ⇒ metadata 必须与用例 1 基线逐元素一致。
 */
int RunCausalCase(aclTensor* cuQ, aclTensor* cuK, const std::vector<int32_t>& cuKHost, int64_t numHeads,
                  int64_t headDim, int64_t metaElems, const std::vector<int64_t>& metaShape, aclrtStream stream,
                  const std::vector<int32_t>& baseline)
{
    const int64_t batch = static_cast<int64_t>(cuKHost.size()) - 1;
    const std::vector<int32_t> ctxHost = {64, 0, 128, 32};
    const std::vector<int32_t> tgtHost = {64, 0, 128, 64};
    constexpr int64_t kTargetGroupSize = 64;
    const std::vector<int64_t> causalShape = {batch};

    std::vector<int32_t> metaHost(static_cast<size_t>(metaElems), kSentinel);
    void* ctxDev = nullptr;
    void* tgtDev = nullptr;
    void* metaDev = nullptr;
    aclTensor* ctxTensor = nullptr;
    aclTensor* tgtTensor = nullptr;
    aclTensor* metaTensor = nullptr;
    CHECK_ACL(CreateAclTensor(ctxHost, causalShape, &ctxDev, aclDataType::ACL_INT32, &ctxTensor),
              "create num_contexts");
    CHECK_ACL(CreateAclTensor(tgtHost, causalShape, &tgtDev, aclDataType::ACL_INT32, &tgtTensor), "create num_targets");
    CHECK_ACL(CreateAclTensor(metaHost, metaShape, &metaDev, aclDataType::ACL_INT32, &metaTensor),
              "create metadata(causal)");

    printf("=== 用例 3：causal mask（mask_mode=1，ctx/tgt/tgs 全下发）===\n");
    if (RunOpAndReadBack(stream, cuQ, cuK, ctxTensor, tgtTensor, numHeads, headDim, kMaskModeCausal, kTargetGroupSize,
                         nullptr, nullptr, metaTensor, metaDev, metaElems, metaHost) != 0) {
        return -1;
    }

    // 覆盖性同样必须成立（行块网格不因 mask 改变）
    if (CheckMetadata(metaHost, metaElems, cuKHost, static_cast<uint32_t>(numHeads), headDim) != 0) {
        printf("[FAIL] causal: metadata 内容校验未通过\n");
        return -1;
    }
    // 权重等价 ⇒ 切分等价：tril 全覆盖数据集下 causal 权重恒 1 == qBlkCnt，应与 no-mask 基线一致
    if (metaHost != baseline) {
        printf("[FAIL] causal 在 tril 全覆盖数据集上应与 no-mask（qBlkCnt=1）切分逐元素一致，实际存在差异\n");
        return -1;
    }
    printf("[PASS] causal 模式跑通：覆盖性校验通过，且与 no-mask 基线逐元素一致\n");

    aclDestroyTensor(ctxTensor);
    aclDestroyTensor(tgtTensor);
    aclDestroyTensor(metaTensor);
    aclrtFree(ctxDev);
    aclrtFree(tgtDev);
    aclrtFree(metaDev);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    if (ret != ACL_SUCCESS) {
        printf("Init acl failed. ERROR: %d\n", (int32_t)ret);
        return ret;
    }

    constexpr int64_t kBatch = 4;
    constexpr int64_t kNumHeads = 16;
    constexpr int64_t kHeadDim = 128;

    // q / k 两条 offsets（长度 = batch + 1），两个用例共用
    const std::vector<int32_t> cuQHost = {0, 128, 256, 384, 512};
    const std::vector<int32_t> cuKHost = {0, 256, 512, 1024, 2048};
    const std::vector<int64_t> seqShape = {kBatch + 1};

    // metadata 容量：sectionNum 上界取 batch*numHeads，再按 4096 元素对齐
    const int64_t rawElems = static_cast<int64_t>(RequiredMetadataElements(kBatch * kNumHeads));
    const int64_t metaElems = (rawElems + 4095) / 4096 * 4096;
    const std::vector<int64_t> metaShape = {metaElems};

    void* cuQDev = nullptr;
    void* cuKDev = nullptr;
    aclTensor* cuQ = nullptr;
    aclTensor* cuK = nullptr;
    CHECK_ACL(CreateAclTensor(cuQHost, seqShape, &cuQDev, aclDataType::ACL_INT32, &cuQ), "create cu_seqlens_q");
    CHECK_ACL(CreateAclTensor(cuKHost, seqShape, &cuKDev, aclDataType::ACL_INT32, &cuK), "create cu_seqlens_kv");

    printf("=== HstuAttnMetadataBackward 上板冒烟用例 ===\n");
    printf("  batch=%ld numHeads=%ld headDim=%ld  metadata elems=%ld\n", (long)kBatch, (long)kNumHeads, (long)kHeadDim,
           (long)metaElems);

    std::vector<int32_t> metaBaseline;
    if (RunNoMaskCase(cuQ, cuK, cuKHost, kNumHeads, kHeadDim, metaElems, metaShape, stream, metaBaseline) != 0) {
        return -1;
    }
    if (RunArbitraryCase(cuQ, cuK, cuKHost, kNumHeads, kHeadDim, metaElems, metaShape, stream, metaBaseline) != 0) {
        return -1;
    }
    if (RunCausalCase(cuQ, cuK, cuKHost, kNumHeads, kHeadDim, metaElems, metaShape, stream, metaBaseline) != 0) {
        return -1;
    }

    aclDestroyTensor(cuQ);
    aclDestroyTensor(cuK);
    aclrtFree(cuQDev);
    aclrtFree(cuKDev);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
