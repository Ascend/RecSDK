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

#include "host_emb_ckpt.h"


using namespace std;
using namespace MxRec;

// remember to comment that the function will take over the control on the mem space

void HostEmbCkpt::SetProcessData(CkptData& processData)
{
    saveHostEmbs = nullptr;
    loadHostEmbs = nullptr;
    saveHostEmbs = processData.hostEmbs;
}

void HostEmbCkpt::GetProcessData(CkptData& processData)
{
    saveHostEmbs = nullptr;
    loadHostEmbs = nullptr;
    LOG_INFO("processData.embHashMaps.empty():{}", processData.embHashMaps.empty());
}

vector<CkptDataType> HostEmbCkpt::GetDataTypes()
{
    return saveDataTypes;
}

vector<string> HostEmbCkpt::GetDirNames()
{
    return fileDirNames;
}

vector<string> HostEmbCkpt::GetEmbNames()
{
    vector<string> embNames;
    for (const auto& item : *saveHostEmbs) {
        embNames.push_back(item.first);
    }
    return embNames;
}

// save info and data
CkptTransData HostEmbCkpt::GetDataset(CkptDataType dataType, string embName)
{
    map<CkptDataType, function<void()>> dataTransMap{
        {CkptDataType::EMB_INFO, [this, embName] { SetEmbInfoTrans(embName); }},
        {CkptDataType::EMB_DATA, [this, embName] { SetEmbDataTrans(embName); }}};

    CleanTransfer();
    dataTransMap.at(dataType)();
    return move(transferData);
}

void HostEmbCkpt::SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData)
{
    LOG_INFO("Parameter dataType:{}, embName:{}, loadedData:{}",
        dataType, embName, loadedData.datasetSize);
}

// load info and data
void HostEmbCkpt::SetDatasetForLoadEmb(CkptDataType dataType, string embName, CkptTransData& loadedData,
                                       CkptData& ckptData)
{
    map<CkptDataType, function<void()>> dataLoadMap{
        {CkptDataType::EMB_INFO, [this, embName, &ckptData] { SetEmbInfo(embName, ckptData); }},
        {CkptDataType::EMB_DATA, [this, embName, &ckptData] { SetEmbData(embName, ckptData); }}};

    CleanTransfer();
    transferData = move(loadedData);
    dataLoadMap.at(dataType)();
}

// save Emb info
void HostEmbCkpt::SetEmbInfoTrans(string embName)
{
    auto embInfoSize = GetEmbInfoSize();
    auto& transArr = transferData.int32Arr;
    const auto& hostEmbInfo = saveHostEmbs->at(embName).hostEmbInfo;

    transArr.reserve(embInfoSize);
    transArr.push_back(hostEmbInfo.sendCount);
    transArr.push_back(hostEmbInfo.extEmbeddingSize);
    transArr.push_back(static_cast<int>(hostEmbInfo.devVocabSize));
    transArr.push_back(static_cast<int>(hostEmbInfo.hostVocabSize));
}

// save Emb data
void HostEmbCkpt::SetEmbDataTrans(string embName)
{
    auto embDataRows = GetEmbDataRows(embName);
    transferData.floatArr.reserve(embDataRows);
    for (auto& item : saveHostEmbs->at(embName).embData) {
        transferData.floatArr.push_back(&item[0]);
    }
}

// load Emb info
void HostEmbCkpt::SetEmbInfo(string embName, CkptData& ckptData)
{
    loadHostEmbs = ckptData.hostEmbs;
    auto& hostEmbInfo = (*loadHostEmbs)[embName].hostEmbInfo;
    const auto& transArr = transferData.int32Arr;

    hostEmbInfo.name = embName;
    hostEmbInfo.sendCount = transArr.at(attribEmbInfoSendCntIdx);
    hostEmbInfo.extEmbeddingSize = transArr.at(attribEmbInfoEmbSizeIdx);
    hostEmbInfo.devVocabSize = static_cast<size_t>(transArr.at(attribEmbInfoDevVocabIdx));
    hostEmbInfo.hostVocabSize = static_cast<size_t>(transArr.at(attribEmbInfoHostVocabIdx));
}

// load Emb data
void HostEmbCkpt::SetEmbData(string embName, CkptData& ckptData) const
{
    LOG_INFO("Parameter embName:{}, ckptData:{}", embName, ckptData.embHashMaps.empty());
}

int HostEmbCkpt::GetEmbInfoSize()
{
    transferData.attribute.push_back(embSveElmtNum);
    transferData.attribute.push_back(fourBytes);
    transferData.datasetSize = embSveElmtNum * fourBytes;
    transferData.attributeSize = transferData.attribute.size() * eightBytes;

    return embSveElmtNum;
}

size_t HostEmbCkpt::GetEmbDataSize(string embName)
{
    auto embDataOuterSize = saveHostEmbs->at(embName).embData.size();
    transferData.attribute.push_back(embDataOuterSize);

    auto embDataInnerSize = saveHostEmbs->at(embName).embData.at(0).size();
    transferData.attribute.push_back(embDataInnerSize);

    transferData.attribute.push_back(fourBytes);

    transferData.datasetSize = embDataOuterSize * embDataInnerSize * fourBytes;
    transferData.attributeSize = transferData.attribute.size() * eightBytes;

    return embDataOuterSize * embDataInnerSize;
}

size_t HostEmbCkpt::GetEmbDataRows(string embName)
{
    auto embDataOuterSize = saveHostEmbs->at(embName).embData.size();
    transferData.attribute.push_back(embDataOuterSize);

    auto embDataInnerSize = saveHostEmbs->at(embName).embData.at(0).size();
    transferData.attribute.push_back(embDataInnerSize);

    transferData.attribute.push_back(fourBytes);

    transferData.datasetSize = embDataInnerSize * fourBytes;
    transferData.attributeSize = transferData.attribute.size() * eightBytes;

    return embDataOuterSize;
}
