/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

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
#include <algorithm>
#include <atomic>
#include <map>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
using namespace tensorflow;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using namespace std;
using namespace chrono;
using OpKernelConstructionPtr = OpKernelConstruction*;
using OpKernelContextPtr = OpKernelContext*;
using InferenceContextPtr = ::tensorflow::shape_inference::InferenceContext*;
namespace {
class CustOps : public OpKernel {
public:
    explicit CustOps(OpKernelConstructionPtr context) : OpKernel(context) {}
    void Compute(OpKernelContextPtr context) override
    {
        std::cout << "Cust Ops not installed!!" << std::endl;
    }
    ~CustOps() override = default;
};
}  // namespace
namespace tensorflow {
REGISTER_OP("UserItemFlashAttention")
    .Input("query: T")
    .Input("key_user: T")
    .Input("value_user: T")
    .Input("mask_len: int32")
    .Input("key_item: key_item_type")
    .Input("value_item: value_item_type")
    .Output("attention_out: T")
    .Attr("T: {float16, float32, bfloat16} = DT_FLOAT")
    .Attr("key_item_type: list({float16, float32, bfloat16}) >= 0 = []")  // 通过属性来标记动态输入个数
    .Attr("value_item_type: list({float16, float32, bfloat16}) >= 0 = []")
    .SetShapeFn([](InferenceContext* c) { return Status::OK(); });
REGISTER_KERNEL_BUILDER(Name("UserItemFlashAttention").Device(DEVICE_CPU), CustOps)
}  // namespace tensorflow
