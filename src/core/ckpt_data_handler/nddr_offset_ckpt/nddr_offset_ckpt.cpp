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

#include "nddr_offset_ckpt.h"


using namespace std;
using namespace MxRec;

void NddrOffsetCkpt::SetProcessData(CkptData& processData)
{
    saveMaxOffset.clear();
    loadMaxOffset.clear();
    saveMaxOffset = std::move(processData.maxOffset);
}

void NddrOffsetCkpt::GetProcessData(CkptData& processData)
{
    processData.maxOffset = std::move(loadMaxOffset);
    saveMaxOffset.clear();
    loadMaxOffset.clear();
}

vector<CkptDataType> NddrOffsetCkpt::GetDataTypes()
{
    return saveDataTypes;
}

vector<string> NddrOffsetCkpt::GetDirNames()
{
    return fileDirNames;
}

vector<string> NddrOffsetCkpt::GetEmbNames()
{
    vector<string> embNames;
    for (const auto& item : as_const(saveMaxOffset)) {
        embNames.push_back(item.first);
    }
    return embNames;
}

CkptTransData NddrOffsetCkpt::GetDataset(CkptDataType dataType, string embName)
{
    CleanTransfer();
    transferData.int32Arr.push_back(static_cast<int>(saveMaxOffset.at(embName)));
    transferData.datasetSize = fourBytes;
    transferData.attribute.push_back(1);
    transferData.attribute.push_back(fourBytes);
    transferData.attributeSize = transferData.attribute.size() * eightBytes;
    LOG_INFO("CkptDataType::EMB_INFO:{}, dataType:{}", CkptDataType::EMB_INFO, dataType);
    return move(transferData);
}

void NddrOffsetCkpt::SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData)
{
    CleanTransfer();
    transferData = move(loadedData);
    loadMaxOffset[embName] = transferData.int32Arr.front();
    LOG_INFO("CkptDataType::EMB_INFO:{}, dataType:{}", CkptDataType::EMB_INFO, dataType);
}
