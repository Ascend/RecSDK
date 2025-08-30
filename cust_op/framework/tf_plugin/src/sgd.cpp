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

#include "mxrec_tf_plugin.h"

namespace tensorflow {
    REGISTER_OP("Sgd")
        .Input("gradient: float32")
        .Input("indices: int32")
        .Input("input_var: float32")
        .Input("lr: float32")
        .Attr("weightDecay: float = 0.0")
        .Output("output_var: float32")
        .SetIsStateful()
        .SetShapeFn(::tensorflow::shape_inference::UnknownShape);
    REGISTER_KERNEL_BUILDER(Name("Sgd").Device(DEVICE_CPU), MxRecTfPlugin::CustOps);
}