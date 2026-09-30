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
 * \file hstu_attn_metadata_backward.h
 * \brief HSTU 反向 metadata 的内存布局（HEAD + FA + FD）。
 *
 * 布局口径与 `hstu_attn_metadata`（前向）保持一致，设备侧
 * `MetadataRowBlockScheduler::flash_meta_layout` 是它的本地镜像：
 *
 *   offset 0                      HEAD        [kHeadStride]                     int32
 *   offset kHeadStride            FA          [sectionNum][kAicCoreNum][kFaStride]
 *   offset kHeadStride+FA 区大小   FD          [sectionNum][kAivCoreNum][kFdStride]
 *
 * FA 每条记录写下本核区间的**两个端点**（只写终点、让设备侧反推起点的写法是错的）：
 *   起点 (bn2Start, mStart)：核 0 取本 section 的起点 (sec.bn2Begin, 0)，其余核取上一核的终点；
 *   终点 (bn2End,   mEnd)  ：本核最后一个行块之后（半开区间右端）。
 * 其中 m 是 M 轴（HSTU 反向即 seqK 轴）的块序号，对应 BLOCK_M 方向的行块下标。
 * 设备侧 `MetadataRowBlockScheduler::LoadSection` 读的正是这 4 个字段，再各自展平：
 *   [Flatten(bn2Start, mStart), Flatten(bn2End, mEnd))
 * 故四个字段缺一不可 —— 漏写 start 会让该核的区间从全局 0 开始。
 *
 * 起点是**逐 section 独立**推的（每个 section 内部从 sec.bn2Begin 起算），不跨 section 累加：
 * 反向是 persist 分核，同一个 section 由全部核重新分一遍，section 之间没有续接关系。
 *
 * 未使用的核槽位（coreId >= 该 section 的 usedCoreNum）保持整条记录全 0，于是
 * Flatten(0, 0) == Flatten(0, 0) ⇒ 区间长度为 0 ⇒ 设备侧 blockCnt == 0、本核空转。
 * 设备侧只依赖「长度为 0」，并不关心空槽位里写的是什么，全 0 是与 ClearMetadata 一致的实现选择。
 *
 * 反向不用 FD（无 stream-K，行块整行做完即止），故 FA 的 s2 字段
 * （kFaS2StartIdx / kFaS2EndIdx）与 kFaFirstFdWsIdx 恒 0、HEAD[kHeadIsFdIdx] 恒 0、
 * FD 区整段恒 0；这些字段只作占位，用于与 hstu_attn_metadata（前向）保持布局同构。
 * 设备侧 `flash_meta_layout` 也只镜像到 kFaMEndIdx，未声明 s2 / FD 下标。
 *
 * 本头文件不依赖 CANN 头，host（examples 回读校验）与 AICPU kernel 共用。
 */

#ifndef HSTU_ATTN_METADATA_BACKWARD_H
#define HSTU_ATTN_METADATA_BACKWARD_H

#include <cstddef>
#include <cstdint>

namespace optiling {
namespace backward {

// =============================================================================
// 布局常量
// =============================================================================
constexpr uint32_t kAicCoreNum = 36U;  // 36: AIC 核数（FA 记录的核数上界）
constexpr uint32_t kAivCoreNum = 72U;  // 72: AIV 核数（FD 记录的核数上界）

constexpr uint32_t kHeadStride = 16U;  // 16: HEAD 段单个 section 的 int32 数
constexpr uint32_t kFaStride = 16U;    // 16: FA 段单核记录的 int32 数
constexpr uint32_t kFdStride = 16U;    // 16: FD 段单核记录的 int32 数

// ---- HEAD 字段下标 ----
constexpr uint32_t kHeadSectionNumIdx = 0U;
constexpr uint32_t kHeadIsFdIdx = 1U;
constexpr uint32_t kHeadMBaseSizeIdx = 2U;
constexpr uint32_t kHeadS2BaseSizeIdx = 3U;

// ---- FA 字段下标 ----
constexpr uint32_t kFaBn2StartIdx = 0U;
constexpr uint32_t kFaMStartIdx = 1U;
constexpr uint32_t kFaS2StartIdx = 2U;
constexpr uint32_t kFaBn2EndIdx = 3U;
constexpr uint32_t kFaMEndIdx = 4U;
constexpr uint32_t kFaS2EndIdx = 5U;
constexpr uint32_t kFaFirstFdWsIdx = 6U;

// ---- FD 字段下标（反向不使用 FD，保留占位以保证布局同构）----
constexpr uint32_t kFdBn2Idx = 0U;
constexpr uint32_t kFdMIdx = 1U;
constexpr uint32_t kFdWsIdx = 2U;
constexpr uint32_t kFdWsNumIdx = 3U;
constexpr uint32_t kFdMStartIdx = 4U;
constexpr uint32_t kFdMNumIdx = 5U;

/*! \brief 把一段 int32 缓冲区解释成 HEAD + FA + FD 三段。 */
struct MetadataBuffer {
    int32_t* head = nullptr;  // [kHeadStride]
    int32_t* fa = nullptr;    // [sectionNum][kAicCoreNum][kFaStride]
    int32_t* fd = nullptr;    // [sectionNum][kAivCoreNum][kFdStride]
    uint32_t sectionNum = 0U;
};

/*! \brief 由裸指针构造布局视图。sectionNum 决定 FA/FD 两段的长度。 */
inline MetadataBuffer MakeMetadataBuffer(void* raw, uint32_t sectionNum)
{
    MetadataBuffer buf{};
    if (raw == nullptr) {
        return buf;
    }
    auto* base = static_cast<int32_t*>(raw);
    buf.sectionNum = sectionNum;
    buf.head = base;
    buf.fa = base + kHeadStride;
    buf.fd = buf.fa + static_cast<size_t>(sectionNum) * kAicCoreNum * kFaStride;
    return buf;
}

/*! \brief 本 metadata 占用的 int32 元素数（供 host 侧核算缓冲区容量）。 */
inline uint32_t RequiredMetadataElements(uint32_t sectionNum)
{
    return kHeadStride + sectionNum * (kAicCoreNum * kFaStride + kAivCoreNum * kFdStride);
}

/*! \brief 整块清零（含 FA / FD 两段）。 */
inline void ClearMetadata(MetadataBuffer& buf)
{
    const size_t total = static_cast<size_t>(RequiredMetadataElements(buf.sectionNum));
    for (size_t i = 0; i < total; ++i) {
        buf.head[i] = 0;
    }
}

inline void SetHead(MetadataBuffer& buf, uint32_t idx, uint32_t value)
{
    buf.head[idx] = static_cast<int32_t>(value);
}

inline int32_t GetHead(const MetadataBuffer& buf, uint32_t idx)
{
    return buf.head[idx];
}

inline void SetFa(MetadataBuffer& buf, uint32_t sec, uint32_t core, uint32_t idx, uint32_t value)
{
    buf.fa[(static_cast<size_t>(sec) * kAicCoreNum + core) * kFaStride + idx] = static_cast<int32_t>(value);
}

inline int32_t GetFa(const MetadataBuffer& buf, uint32_t sec, uint32_t core, uint32_t idx)
{
    return buf.fa[(static_cast<size_t>(sec) * kAicCoreNum + core) * kFaStride + idx];
}

inline void SetFd(MetadataBuffer& buf, uint32_t sec, uint32_t vec, uint32_t idx, uint32_t value)
{
    buf.fd[(static_cast<size_t>(sec) * kAivCoreNum + vec) * kFdStride + idx] = static_cast<int32_t>(value);
}

inline int32_t GetFd(const MetadataBuffer& buf, uint32_t sec, uint32_t vec, uint32_t idx)
{
    return buf.fd[(static_cast<size_t>(sec) * kAivCoreNum + vec) * kFdStride + idx];
}

}  // namespace backward
}  // namespace optiling

#endif  // HSTU_ATTN_METADATA_BACKWARD_H
