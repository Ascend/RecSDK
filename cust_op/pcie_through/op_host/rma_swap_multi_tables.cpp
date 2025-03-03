/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "rma_log.h"
#include "rma_swap_multi_tables_tiling.h"

constexpr int32_t BLOCK_DIM = 48;

namespace optiling {
    constexpr int32_t RMA_DIM_MAX = 2;
    //换入换出各101M，其中1M是用于标志位
    constexpr int32_t RMA_WORK_SPACE_SIZE = 202 * 1024 * 1024;

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        LOG_DEBUG("Rma TilingFunc");
        RmaSwapMultiTablesTilingData tiling;
        context->SetBlockDim(BLOCK_DIM);
        context->SetTilingKey(1);
        //table_a的dim维度
        auto dimNum = context->GetInputShape(2)->GetStorageShape().GetDimNum();
        //等于1的判断可以删掉
        uint64_t dims[RMA_DIM_MAX] = {0};
        if (dimNum == 1) {
            dims[0] = 1;
            dims[1] = context->GetInputShape(2)->GetStorageShape().GetDim(0);
        } else if (dimNum == RMA_DIM_MAX) {
            dims[0] = context->GetInputTensor(1)->GetShapeSize();   // swap out index length
            dims[1] = context->GetInputShape(2)->GetStorageShape().GetDim(1);   // emb dim
        } else {
            LOG_ERROR("Dim-num %d is invalid.", dimNum);
            return ge::GRAPH_FAILED;
        }
        tiling.set_dimValue(dims);
        tiling.set_dimNum(RMA_DIM_MAX);
        //换入的key的数量
        uint64_t size = context->GetInputTensor(0)->GetShapeSize();
        tiling.set_updateLen(size);

        auto attrs = context->GetAttrs();
        if (attrs == nullptr) {
            return ge::GRAPH_FAILED;
        }
        //有效的table数量，[var + slots]
        auto tableNumAttr = attrs->GetInt(0);
        int tableNum = (int)(*tableNumAttr);
        if (tableNum <= 0) {
            LOG_ERROR("Table num: %d is invalid.", tableNum);
            return ge::GRAPH_FAILED;
        }
        tiling.set_tableNum(tableNum);
        //换入的共享内存地址，共享内存在device的地址
        auto gmemAttrIn = attrs->GetStr(1);
        int32_t *shmSwapIn = (int32_t *)(std::stoul(gmemAttrIn));
        if (shmSwapIn == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling.set_shmSwapIn((uint64_t)shmSwapIn);
        //换出的共享内存地址，共享内存在device的地址
        auto gmemAttrOut = attrs->GetStr(2);
        int32_t *shmSwapOut = (int32_t *)(std::stoul(gmemAttrOut));
        if (shmSwapOut == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling.set_shmSwapOut((uint64_t)shmSwapOut);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = RMA_WORK_SPACE_SIZE + sysWorkspaceSize;

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        //output实际没有什么具体的输出，这里给的是标志位，用于判断48个核是不是有不正常的情况，每个核一个标志位判断
        LOG_DEBUG("RmaSwap InferShape.");
        gert::Shape *outputShape = context->GetOutputShape(0);
        if (outputShape == nullptr) {
            LOG_ERROR("Output shape is null.");
            return ge::GRAPH_FAILED;
        }
        outputShape->SetDimNum(1);
        outputShape->SetDim(0, BLOCK_DIM);

        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class RmaSwapMultiTables : public OpDef {
    public:
        explicit RmaSwapMultiTables(const char* name) : OpDef(name)
        {
            this->Input("swap_in_index")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_INT64, ge::DT_INT64, ge::DT_INT64 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("swap_out_index")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_INT64, ge::DT_INT64, ge::DT_INT64 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("table_a")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("table_b")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("table_c")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("table_d")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("table_e")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Input("table_f")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_INT64, ge::DT_INT64, ge::DT_INT64 })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
            this->Attr("table_num").Int();
            this->Attr("shm_swap_in").String();
            this->Attr("shm_swap_out").String();

            this->SetInferShape(ge::InferShape);

            this->AICore().SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend910b");
            this->AICore().AddConfig("ascend910_93");
        }
    };

    OP_ADD(RmaSwapMultiTables);
}
