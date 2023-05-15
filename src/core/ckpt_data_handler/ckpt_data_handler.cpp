/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-12
 */

#include "ckpt_data_handler.h"


using namespace std;
using namespace MxRec;

uint32_t CkptDataHandler::GetDataElmtBytes(CkptDataType dataType)
{
    return dataElmtBytes.at(static_cast<int>(dataType));
}

string CkptDataHandler::GetDataDirName(CkptDataType dataType)
{
    return dataDirNames.at(static_cast<int>(dataType));
}

void CkptDataHandler::CleanTransfer()
{
    transferData.int64Arr.clear();
    transferData.int32Arr.clear();
    transferData.floatArr.clear();
    transferData.attribute.clear();
    transferData.datasetSize = 0;
    transferData.attributeSize = 0;
}