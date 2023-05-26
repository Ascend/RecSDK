/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: constant initializer module
 * Author: MindX SDK
 * Date: 2022/12/22
 */

#ifndef MX_REC_CONSTANT_INITIALIZER_H
#define MX_REC_CONSTANT_INITIALIZER_H

#include <vector>
#include "initializer/initializer.h"

namespace MxRec {
    using std::vector;

    class ConstantInitializer : public Initializer {
    public:
        ConstantInitializer() = default;
        ConstantInitializer(int start, int len, float value);

        ~ConstantInitializer() override {};

        void GenerateData(const float* emb, const int embSize) override;

        int start;
        int len;
        float value;
    };
}

#endif // MX_REC_CONSTANT_INITIALIZER_H
