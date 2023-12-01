/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: dataset eos ops.
 * Author: MindX SDK
 * Create: 2023
 * History: NA
 */

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
        static constexpr const char *const kOutputTypes = "output_types";
        static constexpr const char *const kOutputShapes = "output_shapes";

        explicit EosDatasetOp(OpKernelConstruction *ctx);

    protected:
        void MakeDataset(OpKernelContext *ctx, DatasetBase *input,
                         DatasetBase **output) override;

    private:
        class Dataset;
    }; // class EosDatasetOp
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EOS_DATASET_OP_H_
