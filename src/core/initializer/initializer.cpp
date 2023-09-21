/*
* Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
* Description: initializer module
* Author: MindX SDK
* Date: 2022/12/22
*/
#include "initializer.h"

#include <string>

#include "constant_initializer/constant_initializer.h"
#include "random_normal_initializer/random_normal_initializer.h"
#include "truncated_normal_initializer/truncated_normal_initializer.h"

using namespace MxRec;

ConstantInitializerInfo::ConstantInitializerInfo(float constantValue, float initK)
    : constantValue(constantValue), initK(initK)
{}

NormalInitializerInfo::NormalInitializerInfo(float mean, float stddev, int seed, float initK)
    : mean(mean), stddev(stddev), seed(seed), initK(initK)
{}

InitializeInfo::InitializeInfo(string &name, int start, int len,
                               ConstantInitializerInfo constantInitializerInfo)
    : name(name), start(start), len(len), constantInitializerInfo(constantInitializerInfo)
{
    if (name == "constant_initializer") {
        initializerType = InitializerType::CONSTANT;
        initializer = make_shared<ConstantInitializer>(start, len, constantInitializerInfo.constantValue,
                                                       constantInitializerInfo.initK);
    } else {
        throw invalid_argument("Invalid Initializer Type.");
    }
}

InitializeInfo::InitializeInfo(string &name, int start, int len, NormalInitializerInfo normalInitializerInfo)
    : name(name), start(start), len(len), normalInitializerInfo(normalInitializerInfo)
{
    if (name == "truncated_normal_initializer") {
        initializerType = InitializerType::TRUNCATED_NORMAL;
        initializer = make_shared<TruncatedNormalInitializer>(start, len, normalInitializerInfo);
    } else if (name == "random_normal_initializer") {
        initializerType = InitializerType::RANDOM_NORMAL;
        initializer = make_shared<RandomNormalInitializer>(start, len, normalInitializerInfo);
    } else {
        throw invalid_argument("Invalid Initializer Type.");
    }
}
