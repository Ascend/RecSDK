/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: constant initializer module
 * Author: MindX SDK
 * Date: 2022/12/22
 */

#include "constant_initializer.h"
#include "utils/common.h"

using namespace std;
using namespace MxRec;

ConstantInitializer::ConstantInitializer(int start, int len, float value, float initK)
    : start(start), len(len), value(value)
{
    initParam = initK;
}

void ConstantInitializer::GenerateData(float* const emb, const int embSize)
{
    LOG(INFO) << StringFormat("Device GenerateEmbData ing using Constant Initializer by value %f., start %d, len %d.",
        value, start, len);
    if (len == 0) {
        return;
    }
    if (embSize < (start + len)) {
        LOG(WARNING) << StringFormat(
            "InitializeInfo start %d  + len %d is larger than embedding size %d.", start, len, embSize);
        return;
    }
    std::fill_n(emb + start, len, initParam * value);
}