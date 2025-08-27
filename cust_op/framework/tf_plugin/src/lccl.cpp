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
    REGISTER_OP("LcclAllToAll")
        .Input("send_data: float")
        .Input("send_count_matrix: int64")
        .Input("shape_vec: int32")
        .Input("peer_mem: int64")
        .Attr("rank: int")
        .Attr("rank_size: int")
        .Attr("dim: int")
        .Output("rev_data: float")
        .SetIsStateful()
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            ShapeHandle dataShape;
            TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &dataShape));
            tensorflow::shape_inference::DimensionHandle rows = c->Dim(dataShape, 0);
            int64_t shape1 = c->Value(rows);
            int dim = 0;
            c->GetAttr("dim", &dim);
            c->set_output(0, c->MakeShape({shape1, dim, 1}));
            return Status::OK();
        });
    REGISTER_KERNEL_BUILDER(Name("LcclAllToAll").Device(DEVICE_CPU), MxRecTfPlugin::CustOps);

    REGISTER_OP("LcclGatherAll")
        .Input("emb_table: float")
        .Input("lookup: int32")
        .Input("send_count_matrix: int64")
        .Input("shape_vec: int32")
        .Input("peer_mem: int64")
        .Attr("rank: int")
        .Attr("rank_size: int")
        .Attr("dim: int")
        .Output("rev_data: float")
        .SetIsStateful()
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            int dim = 0;
            c->GetAttr("dim", &dim);
            ShapeHandle dataShape;
            TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(3), 1, &dataShape));
            tensorflow::shape_inference::DimensionHandle rows = c->Dim(dataShape, 0);
            int64_t shape1 = c->Value(rows);
            c->set_output(0, c->MakeShape({shape1, dim, 1}));
            return Status::OK();
        });
    REGISTER_KERNEL_BUILDER(Name("LcclGatherAll").Device(DEVICE_CPU), MxRecTfPlugin::CustOps);

    REGISTER_OP("LcclAllUss")
        .Input("send_data: float")
        .Input("send_count_matrix: int64")
        .Input("shape_vec: int32")
        .Input("peer_mem: int64")
        .Input("restore: int32")
        .Attr("rank: int")
        .Attr("rank_size: int")
        .Attr("dim: int")
        .Output("rev_data: float")
        .SetIsStateful()
        .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
            int dim = 0;
            c->GetAttr("dim", &dim);
            ShapeHandle dataShape;
            TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(2), 1, &dataShape));
            tensorflow::shape_inference::DimensionHandle rows = c->Dim(dataShape, 0);
            int64_t shape1 = c->Value(rows);
            c->set_output(0, c->MakeShape({shape1, dim, 1}));
            return Status::OK();
        });
    REGISTER_KERNEL_BUILDER(Name("LcclAllUss").Device(DEVICE_CPU), MxRecTfPlugin::CustOps);
}