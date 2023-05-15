/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: initializer module
 * Author: MindX SDK
 * Date: 2022/12/22
 */

#ifndef MX_REC_INITIALIZER_H
#define MX_REC_INITIALIZER_H

#include <vector>
namespace MxRec {
    using std::vector;

    class Initializer {
    public:
        Initializer() = default;
        virtual ~Initializer() {};

        virtual void GenerateData(float* emb, int embSize)= 0;
        int start;
        int len;
    };
}

#endif // MX_REC_INITIALIZER_H