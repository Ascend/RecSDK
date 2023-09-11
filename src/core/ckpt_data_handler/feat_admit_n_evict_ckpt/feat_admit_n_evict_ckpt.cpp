/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-22
 */


#include "feat_admit_n_evict_ckpt.h"

using namespace std;
using namespace MxRec;

void FeatAdmitNEvictCkpt::SetProcessData(CkptData& processData)
{
    ClearData();
    if (processData.table2Thresh.empty() || processData.histRec.timestamps.empty() ||
        processData.histRec.historyRecords.empty()) {
        LOG_ERROR("Missing Feature Admit and Evict data");
        throw std::runtime_error("Missing Feature Admit and Evict data");
    }
    saveTable2Thresh = std::move(processData.table2Thresh);
    saveHistRec = std::move(processData.histRec);
}

void FeatAdmitNEvictCkpt::GetProcessData(CkptData& processData)
{
    processData.table2Thresh = std::move(loadTable2Thresh);
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
    for (const auto& item : saveTable2Thresh) {
        embNames.push_back(item.first);
    }
    return embNames;
}

CkptTransData FeatAdmitNEvictCkpt::GetDataset(CkptDataType dataType, string embName)
{
    map<CkptDataType, function<void()>> dataTransMap { { CkptDataType::TABLE_2_THRESH,
        [=] { SetTable2ThreshTrans(embName); } },
        { CkptDataType::HIST_REC, [=] { SetHistRecTrans(embName); } } };

    CleanTransfer();
    dataTransMap.at(dataType)();
    return move(transferData);
}

void FeatAdmitNEvictCkpt::SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData)
{
    map<CkptDataType, function<void()>> dataLoadMap { { CkptDataType::TABLE_2_THRESH,
        [=] { SetTable2Thresh(embName); } },
        { CkptDataType::HIST_REC, [=] { SetHistRec(embName); } } };

    CleanTransfer();
    transferData = move(loadedData);
    dataLoadMap.at(dataType)();
}

void FeatAdmitNEvictCkpt::ClearData()
{
    saveTable2Thresh.clear();
    loadTable2Thresh.clear();
    saveHistRec.timestamps.clear();
    saveHistRec.historyRecords.clear();
    loadHistRec.timestamps.clear();
    loadHistRec.historyRecords.clear();
}

void FeatAdmitNEvictCkpt::SetTable2ThreshTrans(string embName)
{
    auto table2ThreshSize = GetTable2ThreshSize();
    auto& transArr = transferData.int32Arr;
    const auto& table2Thresh = saveTable2Thresh.at(embName);

    transArr.reserve(table2ThreshSize);
    transArr.push_back(table2Thresh.countThreshold);
    transArr.push_back(table2Thresh.timeThreshold);
}

// save
void FeatAdmitNEvictCkpt::SetHistRecTrans(string embName)
{
    if (GetCombineSwitch()) {
        embName = COMBINE_HISTORY_NAME;
    }
    auto histRecSize = GetHistRecSize(embName);
    auto& transArr = transferData.int64Arr;
    const auto& timeStamp = saveHistRec.timestamps.at(embName);
    const auto& histRecs = saveHistRec.historyRecords.at(embName);

    transArr.reserve(histRecSize);

    transArr.push_back(timeStamp);
    for (const auto& histRec : histRecs) {
        transArr.push_back(histRec.first);
        transArr.push_back(static_cast<int64_t>(histRec.second.count));
        transArr.push_back(static_cast<int64_t>(histRec.second.lastTime));
    }
}

void FeatAdmitNEvictCkpt::SetTable2Thresh(string embName)
{
    const auto& transArr = transferData.int32Arr;
    auto& tens2Thresh = loadTable2Thresh[embName];

    tens2Thresh.tableName = embName;
    tens2Thresh.countThreshold = transArr[countThresholdIdx];
    tens2Thresh.timeThreshold = transArr[timeThresholdIdx];
}

// load
void FeatAdmitNEvictCkpt::SetHistRec(string embName)
{
    if (GetCombineSwitch()) {
        embName = COMBINE_HISTORY_NAME;
    }
    const auto& transArr = transferData.int64Arr;
    const auto& attribute = transferData.attribute;
    auto& timestamp = loadHistRec.timestamps[embName];
    auto& histRecs = loadHistRec.historyRecords[embName];

    timestamp = transArr.front();

    size_t featItemInfoTotalSize = attribute.front() * static_cast<size_t>(featItemInfoSaveNum);
    LOG_DEBUG("====Start SetHistRec, name: {}, featItemInfoTotalSize: {}", embName, featItemInfoTotalSize);

    size_t process = 0;
    size_t printPerStep = ((featItemInfoTotalSize / 100) > 0 ? (featItemInfoTotalSize / 100) : 1);
    for (size_t i = featItemInfoOffset; i < featItemInfoTotalSize + featItemInfoOffset; i += featItemInfoSaveNum) {
        process = i % printPerStep;
        if (process == 1) {
            LOG_DEBUG("====in SetHistRec, process : %f",  i/featItemInfoTotalSize);
        }
        auto featureId = transArr[i + featureIdIdxOffset];
        auto count = transArr[i + countIdxOffset];
        auto lastTime = transArr[i + lastTimeIdxOffset];

        histRecs.emplace(featureId, FeatureItemInfo(static_cast<uint32_t>(count), lastTime));
    }
    LOG_DEBUG("====End SetHistRec, name: {}", embName);
}

int FeatAdmitNEvictCkpt::GetTable2ThreshSize()
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
