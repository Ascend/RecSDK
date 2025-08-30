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
    REGISTER_OP("EmbeddingLookupByAddress")
        .Input("address: int64")
        .Attr("embedding_dim: int")
        .Attr("embedding_type: int")
        .Output("y: float")
        .SetIsStateful()
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ShapeHandle addrShape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &addrShape));
            int embSize;
            TF_RETURN_IF_ERROR(c->GetAttr("embedding_dim", &embSize));
            tensorflow::shape_inference::DimensionHandle rows = c->Dim(addrShape, 0);
            c->set_output(0, c->Matrix(rows, embSize));
            return Status::OK();
        });
    REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupByAddress").Device(DEVICE_CPU), MxRecTfPlugin::CustOps);

    REGISTER_OP("EmbeddingUpdateByAddress")
        .Input("address: int64")
        .Input("embedding: float")
        .Attr("update_type: int")
        .Output("y: float")
        .SetIsStateful()
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ShapeHandle addrShape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &addrShape));
            ShapeHandle embeddingShape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &embeddingShape));
            tensorflow::shape_inference::DimensionHandle rows = c->Dim(addrShape, 0);
            tensorflow::shape_inference::DimensionHandle cols = c->Dim(embeddingShape, 1);
            c->set_output(0, c->Matrix(rows, cols));
            return Status::OK();
        });
    REGISTER_KERNEL_BUILDER(Name("EmbeddingUpdateByAddress").Device(DEVICE_CPU), MxRecTfPlugin::CustOps);
}
