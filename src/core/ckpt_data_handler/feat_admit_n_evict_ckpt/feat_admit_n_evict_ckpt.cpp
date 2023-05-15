/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-22
 */

#include <spdlog/spdlog.h>

#include "feat_admit_n_evict_ckpt.h"

using namespace std;
using namespace MxRec;

void FeatAdmitNEvictCkpt::SetProcessData(CkptData& processData)
{
    ClearData();
    if (processData.tens2Thresh.empty() || processData.histRec.timestamps.empty() ||
        processData.histRec.historyRecords.empty()) {
        spdlog::error("Missing Feature Admit and Evict data");
    }
    saveTens2Thresh = std::move(processData.tens2Thresh);
    saveHistRec = std::move(processData.histRec);
}

void FeatAdmitNEvictCkpt::GetProcessData(CkptData& processData)
{
    processData.tens2Thresh = std::move(loadTens2Thresh);
    processData.histRec = std::move(loadHistRec);
    ClearData();
}

vector<CkptDataType> FeatAdmitNEvictCkpt::GetDataTypes()
{
    return saveDataTypes;
}

vector<string> FeatAdmitNEvictCkpt::GetDirNames()
{
    return fileDirNames;
}

vector<string> FeatAdmitNEvictCkpt::GetEmbNames()
{
    vector<string> embNames;
    for (const auto& item : saveTens2Thresh) {
        embNames.push_back(item.first);
    }
    return embNames;
}

CkptTransData FeatAdmitNEvictCkpt::GetDataset(CkptDataType dataType, string embName)
{
    map<CkptDataType, function<void()>> dataTransMap { { CkptDataType::TENSOR_2_THRESH,
        [=] { SetTens2ThreshTrans(embName); } },
        { CkptDataType::HIST_REC, [=] { SetHistRecTrans(embName); } } };

    CleanTransfer();
    dataTransMap.at(dataType)();
    return move(transferData);
}

void FeatAdmitNEvictCkpt::SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData)
{
    map<CkptDataType, function<void()>> dataLoadMap { { CkptDataType::TENSOR_2_THRESH,
        [=] { SetTens2Thresh(embName); } },
        { CkptDataType::HIST_REC, [=] { SetHistRec(embName); } } };

    CleanTransfer();
    transferData = move(loadedData);
    dataLoadMap.at(dataType)();
}

void FeatAdmitNEvictCkpt::ClearData()
{
    saveTens2Thresh.clear();
    loadTens2Thresh.clear();
    saveHistRec.timestamps.clear();
    saveHistRec.historyRecords.clear();
    loadHistRec.timestamps.clear();
    loadHistRec.historyRecords.clear();
}

void FeatAdmitNEvictCkpt::SetTens2ThreshTrans(string embName)
{
    auto tens2ThreshSize = GetTens2ThreshSize();
    auto& transArr = transferData.int32Arr;
    const auto& tens2Thresh = saveTens2Thresh.at(embName);

    transArr.reserve(tens2ThreshSize);
    transArr.push_back(tens2Thresh.countThreshold);
    transArr.push_back(tens2Thresh.timeThreshold);
}

void FeatAdmitNEvictCkpt::SetHistRecTrans(string embName)
{
    auto histRecSize = GetHistRecSize(embName);
    auto& transArr = transferData.int64Arr;
    const auto& timeStamp = saveHistRec.timestamps.at(embName);
    const auto& histRecs = saveHistRec.historyRecords.at(embName);

    transArr.reserve(histRecSize);

    transArr.push_back(timeStamp);
    for (const auto& histRec : histRecs) {
        transArr.push_back(histRec.second.featureId);
        transArr.push_back(static_cast<int64_t>(histRec.second.count));
        transArr.push_back(static_cast<int64_t>(histRec.second.lastTime));
    }
}

void FeatAdmitNEvictCkpt::SetTens2Thresh(string embName)
{
    const auto& transArr = transferData.int32Arr;
    auto& tens2Thresh = loadTens2Thresh[embName];

    tens2Thresh.tensorName = embName;
    tens2Thresh.countThreshold = transArr[countThresholdIdx];
    tens2Thresh.timeThreshold = transArr[timeThresholdIdx];
}

void FeatAdmitNEvictCkpt::SetHistRec(string embName)
{
    const auto& transArr = transferData.int64Arr;
    const auto& attribute = transferData.attribute;
    auto& timestamp = loadHistRec.timestamps[embName];
    auto& histRecs = loadHistRec.historyRecords[embName];

    timestamp = transArr.front();

    size_t featItemInfoTotalSize = attribute.front() * static_cast<size_t>(featItemInfoSaveNum);
    for (size_t i = featItemInfoOffset; i < featItemInfoTotalSize + featItemInfoOffset; i += featItemInfoSaveNum) {
        const auto& featureId = transArr[i + featureIdIdxOffset];
        const auto& count = transArr[i + countIdxOffset];
        const auto& lastTime = transArr[i + lastTimeIdxOffset];

        histRecs[featureId].featureId = featureId;
        histRecs[featureId].count = count;
        histRecs[featureId].lastTime = lastTime;
        histRecs[featureId].tensorName = embName;
    }
}

int FeatAdmitNEvictCkpt::GetTens2ThreshSize()
{
    auto& attribute = transferData.attribute;
    auto& attribSize = transferData.attributeSize;
    auto& datasetSize = transferData.datasetSize;

    attribute.push_back(threshValSaveNum);
    attribute.push_back(fourBytes);
    attribSize = attribute.size() * eightBytes;

    datasetSize = threshValSaveNum * fourBytes;

    return threshValSaveNum;
}

size_t FeatAdmitNEvictCkpt::GetHistRecSize(string embName)
{
    auto& attribute = transferData.attribute;
    auto& attribSize = transferData.attributeSize;
    auto& datasetSize = transferData.datasetSize;

    size_t timeStampNum = 1; // there will be only 1 timeStamp per embName
    auto hashElmtNum = saveHistRec.historyRecords.at(embName).size();
    attribute.push_back(hashElmtNum);
    auto elmtCount = timeStampNum + featItemInfoSaveNum * static_cast<size_t>(hashElmtNum);

    attribute.push_back(eightBytes);
    attribSize = attribute.size() * eightBytes;

    datasetSize = elmtCount * eightBytes;

    return elmtCount;
}
