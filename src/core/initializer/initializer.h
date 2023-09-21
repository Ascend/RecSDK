/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
 * Description: initializer module
 * Author: MindX SDK
 * Date: 2022/12/22
 */

#ifndef MX_REC_INITIALIZER_H
#define MX_REC_INITIALIZER_H

#include <string>
#include <memory>

namespace MxRec {
    using namespace std;

    class Initializer {
    public:
        Initializer() = default;
        virtual ~Initializer() {};

        virtual void GenerateData(float *emb, int embSize) = 0;
        int start;
        int len;
        float initParam = 1.0;
    };

    enum class InitializerType {
        INVALID,
        CONSTANT,
        TRUNCATED_NORMAL,
        RANDOM_NORMAL
    };

    struct ConstantInitializerInfo {
        ConstantInitializerInfo() = default;

        explicit ConstantInitializerInfo(float constantValue, float initK);

        float constantValue;
        float initK = 1.0;
    };

    struct NormalInitializerInfo {
        NormalInitializerInfo() = default;

        NormalInitializerInfo(float mean, float stddev, int seed, float initK);

        float mean;
        float stddev;
        int seed;
        float initK = 1.0;
    };

    struct InitializeInfo {
        InitializeInfo() = default;

        InitializeInfo(string &name, int start, int len, ConstantInitializerInfo constantInitializerInfo);

        InitializeInfo(string &name, int start, int len, NormalInitializerInfo normalInitializerInfo);

        string name;
        int start;
        int len;
        InitializerType initializerType = InitializerType::INVALID;

        ConstantInitializerInfo constantInitializerInfo;
        NormalInitializerInfo normalInitializerInfo;

        std::shared_ptr<Initializer> initializer;
    };
}

#endif // MX_REC_INITIALIZER_H