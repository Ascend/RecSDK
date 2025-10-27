/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

        bool RangeValidate(int begin, int lens);
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