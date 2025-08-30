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
    REGISTER_OP("RmaSwapMultiTables")
        .Input("swap_in_index: int64")
        .Input("swap_out_index: int64")
        .Input("table_a: float32")
        .Input("table_b: float32")
        .Input("table_c: float32")
        .Input("table_d: float32")
        .Input("table_e: float32")
        .Input("table_f: float32")
        .Output("output: int64")
        .Attr("table_num: int")
        .Attr("shm_swap_in: string")
        .Attr("shm_swap_out: string")
        .SetIsStateful()
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->MakeShape({48}));
            return Status::OK();
        });
    REGISTER_KERNEL_BUILDER(Name("RmaSwapMultiTables").Device(DEVICE_CPU), MxRecTfPlugin::CustOps);
}