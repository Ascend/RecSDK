/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-17
 */

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
    for (const auto& item : saveMaxOffset) {
        embNames.push_back(item.first);
    }
    return embNames;
}

CkptTransData NddrOffsetCkpt::GetDataset(CkptDataType dataType, string embName)
{
    CleanTransfer();
    transferData.int32Arr.push_back(saveMaxOffset.at(embName));
    transferData.datasetSize = fourBytes;
    transferData.attribute.push_back(1);
    transferData.attribute.push_back(fourBytes);
    transferData.attributeSize = transferData.attribute.size() * eightBytes;
    return move(transferData);
}

void NddrOffsetCkpt::SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData)
{
    CleanTransfer();
    transferData = move(loadedData);
    loadMaxOffset[embName] = transferData.int32Arr.front();
}