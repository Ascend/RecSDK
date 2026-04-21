/* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

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

/**
 * @file hstu_backward_v2_define.h
 * @brief HSTU (Hierarchical Sparse Transformer Unit) Backward V2 算子定义头文件
 * @description 定义算子使用的维度索引、输入输出索引和属性索引枚举
 */

#ifndef HSTU_BACKWARD_V2_DEFINE_H
#define HSTU_BACKWARD_V2_DEFINE_H

/**
 * @brief 维度索引枚举
 * @description 用于指定张量的维度顺序：ZERO=批次维度, ONE=头数维度, TWO=序列维度, THREE=特征维度
 */
enum class DIM_INDEX : uint32_t { ZERO = 0, ONE, TWO, THREE };

/**
 * @brief 输入张量索引枚举
 * @description 定义算子输入参数的索引顺序：
 * - GRAD: 反向传播的梯度输入
 * - Q: Query 张量
 * - K: Key 张量
 * - V: Value 张量
 * - RAB: 相对位置注意力偏置 (Relative Attention Bias)
 * - SEQ_OFFSET_Q: Query 的序列偏移量
 * - SEQ_OFFSET_K: Key/Value 的序列偏移量
 * - NUM_CONTEXT: 上下文序列长度
 * - NUM_TARGET: 目标序列长度
 */
enum class IN_INDEX : uint32_t { GRAD = 0, Q, K, V, RAB, SEQ_OFFSET_Q, SEQ_OFFSET_K, NUM_CONTEXT, NUM_TARGET };

/**
 * @brief 输出张量索引枚举
 * @description 定义算子输出参数的索引顺序：
 * - Q_GRAD: Query 的梯度
 * - K_GRAD: Key 的梯度
 * - V_GRAD: Value 的梯度
 * - RAB_GRAD: 相对位置注意力偏置的梯度
 */
enum class OUT_INDEX : uint32_t { Q_GRAD = 0, K_GRAD, V_GRAD, RAB_GRAD };

/**
 * @brief 属性索引枚举
 * @description 定义算子属性的索引顺序：
 * - MAX_SEQLEN_Q: Query 的最大序列长度
 * - MAX_SEQLEN_K: Key/Value 的最大序列长度
 * - SCALE: 缩放因子
 * - TARGET_GROUP_SIZE: 目标分组大小
 * - ALPHA: 注意力分数的缩放系数
 */
enum class ATTR_INDEX : uint32_t { MAX_SEQLEN_Q = 0, MAX_SEQLEN_K = 1, SCALE, TARGET_GROUP_SIZE, ALPHA };

#endif