/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-12
 */

#include "host_emb_ckpt.h"


using namespace std;
using namespace MxRec;

// remember to comment that the function will take over the control on the mem space

void HostEmbCkpt::SetProcessData(CkptData& processData)
{
    saveHostEmbs.clear();
    loadHostEmbs.clear();
    saveHostEmbs = std::move(processData.hostEmbs);
}

void HostEmbCkpt::GetProcessData(CkptData& processData)
{
    processData.hostEmbs = std::move(loadHostEmbs);
    saveHostEmbs.clear();
    loadHostEmbs.clear();
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
    for (const auto& item : saveHostEmbs) {
        embNames.push_back(item.first);
    }
    return embNames;
}

CkptTransData HostEmbCkpt::GetDataset(CkptDataType dataType, string embName)
{
    map<CkptDataType, function<void()>> dataTransMap { { CkptDataType::EMB_INFO, [=] { SetEmbInfoTrans(embName); } },
        { CkptDataType::EMB_DATA, [=] { SetEmbDataTrans(embName); } } };

    CleanTransfer();
    dataTransMap.at(dataType)();
    return move(transferData);
}

void HostEmbCkpt::SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData)
{
    map<CkptDataType, function<void()>> dataLoadMap { { CkptDataType::EMB_INFO, [=] { SetEmbInfo(embName); } },
        { CkptDataType::EMB_DATA, [=] { SetEmbData(embName); } } };

    CleanTransfer();
    transferData = move(loadedData);
    dataLoadMap.at(dataType)();
}

void HostEmbCkpt::SetEmbInfoTrans(string embName)
{
    auto embInfoSize = GetEmbInfoSize();
    auto& transArr = transferData.int32Arr;
    const auto& hostEmbInfo = saveHostEmbs.at(embName).hostEmbInfo;

    transArr.reserve(embInfoSize);
    transArr.push_back(hostEmbInfo.sendCount);
    transArr.push_back(hostEmbInfo.embeddingSize);
    transArr.push_back(static_cast<int>(hostEmbInfo.devVocabSize));
    transArr.push_back(static_cast<int>(hostEmbInfo.hostVocabSize));
}

void HostEmbCkpt::SetEmbDataTrans(string embName)
{
    auto embDataSize = GetEmbDataSize(embName);
    transferData.floatArr.reserve(embDataSize);
    for (const auto& item : saveHostEmbs.at(embName).embData) {
        transferData.floatArr.insert(transferData.floatArr.end(), item.begin(), item.end());
    }
}

void HostEmbCkpt::SetEmbInfo(string embName)
{
    auto& hostEmbInfo = loadHostEmbs[embName].hostEmbInfo;
    const auto& transArr = transferData.int32Arr;

    hostEmbInfo.name = embName;
    hostEmbInfo.sendCount = transArr.at(attribEmbInfoSendCntIdx);
    hostEmbInfo.embeddingSize = transArr.at(attribEmbInfoEmbSizeIdx);
    hostEmbInfo.devVocabSize = static_cast<size_t>(transArr.at(attribEmbInfoDevVocabIdx));
    hostEmbInfo.hostVocabSize = static_cast<size_t>(transArr.at(attribEmbInfoHostVocabIdx));
}

void HostEmbCkpt::SetEmbData(string embName)
{
    vector<float> embValues;
    auto embDataOuterSize = transferData.attribute.at(attribEmbDataOuterIdx);
    auto embDataInnerSize = transferData.attribute.at(attribEmbDataInnerIdx);
    auto rawBegin = transferData.floatArr.begin();
    loadHostEmbs[embName].embData.reserve(embDataOuterSize);
    for (size_t i = 0; i < embDataOuterSize; ++i) {
        size_t beginShift = i * embDataInnerSize;
        size_t endShift = (i + 1) * embDataInnerSize;
        embValues.reserve(embDataInnerSize);
        embValues.insert(embValues.begin(), rawBegin + beginShift, rawBegin + endShift);
        loadHostEmbs[embName].embData.push_back(move(embValues));
    }
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
    auto embDataOuterSize = saveHostEmbs.at(embName).embData.size();
    transferData.attribute.push_back(embDataOuterSize);

    auto embDataInnerSize = saveHostEmbs.at(embName).embData.at(0).size();
    transferData.attribute.push_back(embDataInnerSize);

    transferData.attribute.push_back(fourBytes);

    transferData.datasetSize = embDataOuterSize * embDataInnerSize * fourBytes;
    transferData.attributeSize = transferData.attribute.size() * eightBytes;

    return embDataOuterSize * embDataInnerSize;
}