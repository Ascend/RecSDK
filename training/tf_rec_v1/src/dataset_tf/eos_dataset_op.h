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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_EOS_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EOS_DATASET_OP_H_

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/public/version.h"

#if TF_MAJOR_VERSION == 2
#define TF_VERSION_TF2
#endif

namespace tensorflow {
namespace data {
    // 这个类的 MakeDataset() 方法告诉 TensorFlow 怎样根据一个操作的输入和属性生成一个数据集的对象。
    class EosDatasetOp : public UnaryDatasetOpKernel {
    public:
        static constexpr const char *const kDatasetType = "Eos";
        static constexpr const char *const kInputDataset = "input_dataset";
        static constexpr const char *const kChannelId = "channel_id";
        static constexpr const char *const kMaxTrainSteps = "max_train_steps";
        static constexpr const char *const kMaxEvalSteps = "max_eval_steps";
        static constexpr const char *const kOutputTypes = "output_types";
        static constexpr const char *const kOutputShapes = "output_shapes";

        explicit EosDatasetOp(OpKernelConstruction *ctx);

    protected:
        void MakeDataset(OpKernelContext *ctx, DatasetBase *input,
                         DatasetBase **output) override;

    private:
        class Dataset;
        DataTypeVector outputTypes_;
        std::vector<PartialTensorShape> outputShapes_;
    }; // class EosDatasetOp
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EOS_DATASET_OP_H_
