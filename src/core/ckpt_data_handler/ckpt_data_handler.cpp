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

#include "ckpt_data_handler.h"


using namespace std;
using namespace MxRec;

uint32_t CkptDataHandler::GetDataElmtBytes(CkptDataType dataType)
{
    return dataTypeInfoMap.at(dataType).second;
}

string CkptDataHandler::GetDataDirName(CkptDataType dataType)
{
    return dataTypeInfoMap.at(dataType).first;
}

void CkptDataHandler::CleanTransfer()
{
    transferData.int64Arr.clear();
    transferData.int32Arr.clear();
    transferData.attribute.clear();
    transferData.datasetSize = 0;
    transferData.attributeSize = 0;
}

void CkptDataHandler::SetDatasetForLoadEmb(CkptDataType dataType, string embName, CkptTransData& loadedData,
                                           CkptData& ckptData)
{
    string errMsg = Logger::Format("Load host emb failed, dataType:{}, embName:{}, loadedData:{}."
                                   " Only EMB_INFO and EMB_DATA supported for load host emb.",
                                   dataType, embName, loadedData.datasetSize);
    auto error = Error(ModuleName::M_CHECK_POINT, ErrorType::NOT_SUPPORTED, errMsg);
    LOG_ERROR(error.ToString());
    throw std::runtime_error(error.ToString());
}