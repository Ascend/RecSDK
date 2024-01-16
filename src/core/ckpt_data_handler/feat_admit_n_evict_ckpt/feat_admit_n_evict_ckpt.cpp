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
    map<CkptDataType, function<void()>> dataTransMap{
        {CkptDataType::TABLE_2_THRESH, [this, embName] { SetTable2ThreshTrans(embName); }},
        {CkptDataType::HIST_REC,       [this, embName] { SetHistRecTrans(embName); }}};

    CleanTransfer();
    dataTransMap.at(dataType)();
    return move(transferData);
}

void FeatAdmitNEvictCkpt::SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData)
{
    map<CkptDataType, function<void()>> dataLoadMap{
        {CkptDataType::TABLE_2_THRESH, [this, embName] { SetTable2Thresh(embName); }},
        {CkptDataType::HIST_REC,       [this, embName] { SetHistRec(embName); }}};

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
    int32_t isSum = table2Thresh.isEnableSum ? 1 : 0;
    transArr.push_back(isSum);
}

// save
void FeatAdmitNEvictCkpt::SetHistRecTrans(string embName)
{
    if (GlobalEnv::useCombineFaae) {
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
    tens2Thresh.isEnableSum = (transArr[isSumThresholdIdx] == 1);
}

// load
void FeatAdmitNEvictCkpt::SetHistRec(string embName)
{
    if (GlobalEnv::useCombineFaae) {
        embName = COMBINE_HISTORY_NAME;
    }
    const auto& transArr = transferData.int64Arr;
    const auto& attribute = transferData.attribute;
    auto& timestamp = loadHistRec.timestamps[embName];
    auto& histRecs = loadHistRec.historyRecords[embName];
    if (transArr.empty() || attribute.empty()) {
        throw std::runtime_error("transArr or attribute is empty");
    }
    timestamp = transArr.front();

    size_t featItemInfoTotalSize = attribute.front() * static_cast<size_t>(featItemInfoSaveNum);
    LOG_DEBUG("====Start SetHistRec, name: {}, featItemInfoTotalSize: {}", embName, featItemInfoTotalSize);

    size_t process = 0;
    size_t printPerStep = ((featItemInfoTotalSize / 100) > 0 ? (featItemInfoTotalSize / 100) : 1);
    for (size_t i = featItemInfoOffset; i < featItemInfoTotalSize + featItemInfoOffset; i += featItemInfoSaveNum) {
        process = i % printPerStep;
        if (process == 1) {
            LOG_DEBUG("====in SetHistRec, process : %f", i / featItemInfoTotalSize);
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
