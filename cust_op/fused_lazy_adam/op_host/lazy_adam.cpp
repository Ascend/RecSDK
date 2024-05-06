/* Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.

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

#include "lazy_adam_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
    constexpr int BLOCK_SIZE = 32;
    constexpr int RESERVE_UB_SIZE = 20 * 1024;
    constexpr int DATA_NUM_PER_COMPUTE = 8;
    constexpr int32_t USR_SIZE = 256;
    constexpr int32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;

    template<typename T>
    static ge::graphStatus CheckNullPointer(T* pointer, const char* errorMessage)
    {
        if (pointer == nullptr) {
            printf("%s nullptr\n", errorMessage);
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus LazyAdamTilingFunc(gert::TilingContext* context)
    {
        size_t* currentWorkspace = context->GetWorkspaceSizes(1);
        if (CheckNullPointer(currentWorkspace, "currentWorkspace") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = SYS_WORKSPACE_SIZE + USR_SIZE;

        LazyAdamTilingData tiling;
        const gert::StorageShape* indicesShape = context->GetInputShape(1);
        const gert::StorageShape* inputMShape = context->GetInputShape(2);
        uint64_t dim0 = inputMShape->GetStorageShape().GetDim(0);
        uint64_t dim1 = indicesShape->GetStorageShape().GetDim(0);
        uint64_t dim2 = inputMShape->GetStorageShape().GetDim(1);
        ge::DataType inputMDtype = context->GetInputDesc(2)->GetDataType();
        int inputMDtypeSize = ge::GetSizeByDataType(inputMDtype);
        ge::DataType indicesDtype = context->GetInputDesc(1)->GetDataType();
        int indicesDtypeSize = ge::GetSizeByDataType(indicesDtype);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        auto attrs = context->GetAttrs();

        float beta1 = *attrs->GetAttrPointer<float>(0);
        float beta2 = *attrs->GetAttrPointer<float>(1);
        float epsilon = *attrs->GetAttrPointer<float>(2);

        auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = platformInfo.GetCoreNum();
        uint64_t ub;
        platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        ub = ub - RESERVE_UB_SIZE;
        // ub大小除以每行的数据大小，得到每次处理的行数
        uint64_t row = ub / (dim2 * inputMDtypeSize * DATA_NUM_PER_COMPUTE + 1 * indicesDtypeSize);
        if (row > dim1) {
            row = dim1;
        }

        // 保证申请的内存是32的倍数并且向上取整 计算方式：(num+31)/32*32
        uint64_t indicesAllocSize = (row * indicesDtypeSize + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        uint64_t otherAllocSize = (row * inputMDtypeSize * dim2 + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        // 前 CORE_NUM - 1 个核分配的任务量
        uint64_t batch = dim1 / coreNum;
        // 实际使用的核数
        context->SetBlockDim(coreNum);
        uint64_t loopCount = batch / row;  // CORE_NUM - 1 个核的任务量，除以UB每一次能处理的数据，得到处理次数
        uint64_t rowLeft = batch - row * loopCount;  // UB处理 loopCount 那么多次后，分给当前core剩下的数据量

        // 最后一个核分配的任务量
        uint64_t batchTail = dim1 - batch * (coreNum - 1);  // phy 该写法适配了dim1刚好整除coreNum的情况
        uint64_t loopCountTail = batchTail / row;
        uint64_t rowLeftTail = batchTail - row * loopCountTail;

        tiling.set_beta1(beta1);
        tiling.set_beta2(beta2);
        tiling.set_epsilon(epsilon);
        tiling.set_dim0(dim0);
        tiling.set_dim1(dim1);
        tiling.set_dim2(dim2);
        tiling.set_row(row);  // 每个ai core一次能分配的数据行数
        tiling.set_indicesAllocSize(indicesAllocSize);  // indices大小，用于申请空间
        tiling.set_otherAllocSize(otherAllocSize);  // 入参中非indices要申请的空间大小
        tiling.set_batch(batch);  // 前CORE_NUM - 1个核分配的任务量
        tiling.set_loopCount(loopCount);  // 前CORE_NUM - 1 个核内循环处理次数
        tiling.set_rowLeft(rowLeft);  // 前CORE_NUM - 1 个核, 核内处理 loopCount 次后，分给当前core剩下的数据量
        tiling.set_loopCountTail(loopCountTail);  // 最后一个核，核内循环次数
        tiling.set_rowLeftTail(rowLeftTail);  // 最后一个核，核内循环loopCountTail次后，剩余数据量
        tiling.set_coreNum(coreNum);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static ge::graphStatus LazyAdamInferShape(gert::InferShapeContext* context)
    {
        if (optiling::CheckNullPointer(context, "context") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }

        gert::Shape* outputMShape = context->GetOutputShape(0);
        if (optiling::CheckNullPointer(outputMShape, "outputMShape") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape* inputMShape = context->GetInputShape(2);
        if (optiling::CheckNullPointer(inputMShape, "inputMShape") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        *outputMShape = *inputMShape;

        gert::Shape* outputVShape = context->GetOutputShape(1);
        if (optiling::CheckNullPointer(outputVShape, "outputVShape") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape* inputVShape = context->GetInputShape(3);
        if (optiling::CheckNullPointer(inputVShape, "inputVShape") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        *outputVShape = *inputVShape;

        gert::Shape* outputVarShape = context->GetOutputShape(2);
        if (optiling::CheckNullPointer(outputVarShape, "outputVarShape") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape* inputVarShape = context->GetInputShape(4);
        if (optiling::CheckNullPointer(inputVarShape, "inputVarShape") != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        *outputVarShape = *inputVarShape;

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus LazyAdamInferDataType(gert::InferDataTypeContext* context)
    {
        return GRAPH_SUCCESS;
    }
}


namespace ops {
    class LazyAdam : public OpDef {
    public:
        explicit LazyAdam(const char* name) : OpDef(name)
        {
            this->Input("gradient")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("indices")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_INT32})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("inputM")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("inputV")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("inputVar")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Input("lr")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("inputM")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("inputV")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Output("inputVar")
                    .ParamType(REQUIRED)
                    .DataType({ge::DT_FLOAT})
                    .Format({ge::FORMAT_ND})
                    .UnknownShapeFormat({ge::FORMAT_ND});
            this->Attr("beta1").Float();
            this->Attr("beta2").Float();
            this->Attr("epsilon").Float();
            this->SetInferShape(ge::LazyAdamInferShape)
                    .SetInferDataType(ge::LazyAdamInferDataType);

            this->AICore().SetTiling(optiling::LazyAdamTilingFunc);
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(LazyAdam);
}
