/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: dataset eos ops.
 * Author: MindX SDK
 * Create: 2023
 * History: NA
 */

#include "eos_dataset_op.h"

#include <mpi.h>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/mutex.h"
#if defined(TF_VERSION_TF2)
#include "tensorflow/core/data/name_utils.h"
#endif

#include "key_process/key_process.h"
#include "utils/logger.h"

using namespace std;
using namespace MxRec;

namespace tensorflow {
namespace data {

MPI_Comm g_comm;
MPI_Group g_worldGroup;

constexpr const char *const EosDatasetOp::kDatasetType;
constexpr const char *const EosDatasetOp::kInputDataset;
constexpr const char *const EosDatasetOp::kChannelId;
constexpr const char *const EosDatasetOp::kOutputTypes;
constexpr const char *const EosDatasetOp::kOutputShapes;

// 表示数据集的不可变性定义，这个类的 MakeIterator() 方法告诉 TensorFlow 怎样在数据集上生成迭代器对象。
class EosDatasetOp::Dataset : public DatasetBase {
public:
    explicit Dataset(OpKernelContext *ctx, const DatasetBase *input, int32_t channelId)
        : DatasetBase(DatasetContext(ctx)),
        input_(input),
        channelId_(channelId)
    {
        input_->Ref();
        auto os_input = input->output_shapes();
        output_shapes_ = os_input;
        keyProcess = Singleton<KeyProcess>::GetInstance();
        MPI_Comm_group(MPI_COMM_WORLD, &g_worldGroup);
        MPI_Comm_create(MPI_COMM_WORLD, g_worldGroup, &g_comm);
    }

    ~Dataset() override
    {
        input_->Unref();
    }

    std::unique_ptr <IteratorBase> MakeIteratorInternal(const string &prefix) const override
    {
#if defined(TF_VERSION_TF2)
        string prefix_para = name_utils::IteratorPrefix(kDatasetType, prefix);
#else
        string prefix_para = prefix + "::" + kDatasetType;
#endif
        return absl::make_unique<Iterator>(Iterator::Params{
                this, prefix_para});
    }

    const DataTypeVector &output_dtypes() const override
    {
        return input_->output_dtypes();
    }

    const std::vector <PartialTensorShape> &output_shapes() const override
    {
        return output_shapes_;
    }

    string DebugString() const override
    {
#if defined(TF_VERSION_TF2)
        return name_utils::DatasetDebugString(kDatasetType);
#else
        return "NpuMapDatasetOp::DataSet";
#endif
    }

    int64 Cardinality() const override
    {
        return input_->Cardinality();
    }

    Status CheckExternalState() const override
    {
        return input_->CheckExternalState();
    }

protected:
    Status AsGraphDefInternal(SerializationContext *ctx, DatasetGraphDefBuilder *b, Node **output) const override
    {
        Node *input_graph = nullptr;
        TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph));
        Node *channel_id_x = nullptr;
        TF_RETURN_IF_ERROR(b->AddScalar(channelId_, &channel_id_x));
        TF_RETURN_IF_ERROR(b->AddDataset(this, {input_graph, channel_id_x}, output));
        return Status::OK();
    }

private:
    // 表示特定数据集上的迭代器的可变性，这个类的 GetNextInternal() 方法告诉 TensorFlow 怎样获取迭代器的下一个元素。
    class Iterator : public DatasetIterator<Dataset> {
    public:
        explicit Iterator(const Params &params) : DatasetIterator<Dataset>(params), i_(0) {}
#if defined(TF_VERSION_TF2)
        Status Initialize(IteratorContext* ctx) override
        {
            return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
        }
#else
        Status Initialize(IteratorContext *ctx) override
        {
            return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
        }
#endif
        Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override
        {
            mutex_lock l(mu_);
            int exitFlag = 0;
            if (!input_impl_) {
                *end_of_sequence = true;
                return Status::OK();
            }

            TF_RETURN_IF_ERROR(input_impl_->GetNext(ctx, out_tensors, end_of_sequence));

            // 正常数据流程
            if (!*end_of_sequence) {
                LOG_TRACE("GetNext, step in MPI_Allreduce, exitFlag:[{}]", exitFlag);
                MPI_Allreduce(&exitFlag, &exitFlag, 1, MPI_INT, MPI_SUM, g_comm);
                LOG_TRACE("GetNext, step out MPI_Allreduce, exitFlag:[{}]", exitFlag);
                // 数据不均衡场景, 别的卡eos
                if (exitFlag != 0) {
                    i_ = 1;
                    *end_of_sequence = true;
                    LOG_INFO("GetNext, some rank eos, channelID:[{}]", dataset()->channelId_);
                    dataset()->keyProcess->SetEos(1, dataset()->channelId_);
                }
                return Status::OK();
            }
            // 数据eos场景
            i_ = 1;
            exitFlag = 1;
            *end_of_sequence = true;
            input_impl_.reset();
            LOG_TRACE("GetNext eos, step in MPI_Allreduce, exitFlag:[{}]", exitFlag);
            MPI_Allreduce(&exitFlag, &exitFlag, 1, MPI_INT, MPI_SUM, g_comm);
            LOG_TRACE("GetNext eos, step out MPI_Allreduce, exitFlag:[{}]", exitFlag);

            LOG_INFO("GetNext eos, channelID:[{}]", dataset()->channelId_);
            dataset()->keyProcess->SetEos(1, dataset()->channelId_);

            return Status::OK();
        }
    protected:
        std::shared_ptr<model::Node> CreateNode(
                IteratorContext* ctx, model::Node::Args args) const override
        {
            return model::MakeKnownRatioNode(std::move(args), /* ratio= */ 1);
        }
#if defined(TF_VERSION_TF2)
        Status SaveInternal(SerializationContext* ctx, IteratorStateWriter* writer) override
        {
            TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
            return Status::OK();
        }
#else
        Status SaveInternal(IteratorStateWriter* writer) override
        {
            TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
            return Status::OK();
        }
#endif
        Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override
        {
            mutex_lock l(mu_);
            TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
            return Status::OK();
        }

    private:
        tensorflow::mutex mu_;
        int64 i_ GUARDED_BY(mu_);
        std::unique_ptr <IteratorBase> input_impl_ GUARDED_BY(mu_);
    };
    const DatasetBase *input_;
    int32_t channelId_;
    KeyProcess* keyProcess;
    std::vector <PartialTensorShape> output_shapes_;
};

EosDatasetOp::EosDatasetOp(OpKernelConstruction *ctx) : UnaryDatasetOpKernel(ctx) {}

void EosDatasetOp::MakeDataset(OpKernelContext *ctx, DatasetBase *input, DatasetBase **output)
{
    int32_t channel;
    OP_REQUIRES_OK(ctx, ParseScalarArgument<int32_t>(ctx, kChannelId, &channel));
    *output = new Dataset(ctx, input, channel);
}

REGISTER_OP("EosDataset")
.Input("input_dataset: variant")
.Input("channel_id: int32")
.Output("handle: variant")
.Attr("output_types: list(type) >= 1")
.Attr("output_shapes: list(shape) >= 1")
.SetShapeFn(shape_inference::ScalarShape);
REGISTER_KERNEL_BUILDER(Name("EosDataset").Device(DEVICE_CPU),
                        EosDatasetOp);

}  // namespace data
}  // namespace tensorflow