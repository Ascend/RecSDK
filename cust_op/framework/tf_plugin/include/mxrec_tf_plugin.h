/* Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;
using namespace std;
using namespace chrono;

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

using OpKernelConstructionPtr = OpKernelConstruction*;
using OpKernelContextPtr = OpKernelContext*;

namespace MxRecTfPlugin {
    class CustOps : public OpKernel {
    public:
        explicit CustOps(OpKernelConstructionPtr context) : OpKernel(context)
        {
        }

        void Compute(OpKernelContextPtr context) override
        {
            std::cout << "context " << context->step_id() << std::endl;
            std::cout << " Cust opp not installed!!" << std::endl;
        }

        ~CustOps() override = default;
    };
}
