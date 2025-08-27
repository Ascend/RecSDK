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
    REGISTER_OP("AttentionFusion")
        .Input("query: float")
        .Input("key: float")
        .Input("value: float")
        .Input("atten_mask: float")
        .Output("atten_score: float")
        .Output("softmax_out: float")
        .Attr("mask_on: int")
        .SetIsStateful()
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ShapeHandle query_shape;
            ShapeHandle key_shape;
            ShapeHandle value_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &query_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &key_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &value_shape));

            tensorflow::shape_inference::DimensionHandle queryDim0 = c->Dim(query_shape, 0);
            tensorflow::shape_inference::DimensionHandle queryDim1 = c->Dim(query_shape, 1);
            tensorflow::shape_inference::DimensionHandle keyDim1 = c->Dim(key_shape, 1);
            tensorflow::shape_inference::DimensionHandle valueDim2 = c->Dim(value_shape, 2);
            int64_t shape0 = c->Value(queryDim0);
            int64_t shape1 = c->Value(queryDim1);
            int64_t shape2 = c->Value(keyDim1);
            int64_t shape3 = c->Value(valueDim2);

            c->set_output(0, c->MakeShape({shape0, shape1, shape3}));
            c->set_output(1, c->MakeShape({shape0, shape1, shape2}));
            return Status::OK();
        });
    REGISTER_KERNEL_BUILDER(Name("AttentionFusion").Device(DEVICE_CPU), MxRecTfPlugin::CustOps)

    REGISTER_OP("AttentionFusionGrad")
        .Input("dout: float")
        .Input("softmax_out: float")
        .Input("query: float")
        .Input("key: float")
        .Input("value: float")
        .Output("grad_query: float")
        .Output("grad_key: float")
        .Output("grad_value: float")
        .SetIsStateful()
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ShapeHandle query_shape;
            ShapeHandle key_shape;
            ShapeHandle value_shape;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 3, &query_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 3, &key_shape));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 3, &value_shape));

            tensorflow::shape_inference::DimensionHandle queryDim0 = c->Dim(query_shape, 0);
            tensorflow::shape_inference::DimensionHandle queryDim1 = c->Dim(query_shape, 1);
            tensorflow::shape_inference::DimensionHandle queryDim2 = c->Dim(query_shape, 2);
            tensorflow::shape_inference::DimensionHandle keyDim1 = c->Dim(key_shape, 1);
            tensorflow::shape_inference::DimensionHandle keyDim2 = c->Dim(key_shape, 2);
            tensorflow::shape_inference::DimensionHandle valueDim1 = c->Dim(value_shape, 1);
            tensorflow::shape_inference::DimensionHandle valueDim2 = c->Dim(value_shape, 2);

            int64_t qShape0 = c->Value(queryDim0);
            int64_t qShape1 = c->Value(queryDim1);
            int64_t qShape2 = c->Value(queryDim2);

            int64_t kShape1 = c->Value(keyDim1);
            int64_t kShape2 = c->Value(keyDim2);

            int64_t vShape1 = c->Value(valueDim1);
            int64_t vShape2 = c->Value(valueDim2);

            c->set_output(0, c->MakeShape({qShape0, qShape1, qShape2}));
            c->set_output(1, c->MakeShape({qShape0, kShape1, kShape2}));
            c->set_output(2, c->MakeShape({qShape0, vShape1, vShape2}));
            return Status::OK();
        });
    REGISTER_KERNEL_BUILDER(Name("AttentionFusionGrad").Device(DEVICE_CPU), MxRecTfPlugin::CustOps)
}  // namespace tensorflow