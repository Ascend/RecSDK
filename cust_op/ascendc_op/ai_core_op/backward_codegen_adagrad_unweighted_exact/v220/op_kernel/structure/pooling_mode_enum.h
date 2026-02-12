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

#ifndef POOLING_MODE_ENUM_H
#define POOLING_MODE_ENUM_H

// Pooling 模式枚举（与 PyTorch EmbeddingBag 对齐，与kernel代码保持一致）
enum class PoolingMode {
    POOL_MODE_SUM = 0,  // SUM_POOL - 梯度均匀分发（反向时每个 ID 加相同 grad）
    POOL_MODE_MEAN = 1, // MEAN_POOL - 梯度按 bag size 归一化（反向时每个 ID 加 grad / bag_size）
    POOL_MODE_NONE = 2, // NONE_POOL - 无池化操作
};

#endif // POOLING_MODE_ENUM_H