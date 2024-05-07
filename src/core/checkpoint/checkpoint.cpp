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

#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <omp.h>

#include "ckpt_data_handler/feat_admit_n_evict_ckpt/feat_admit_n_evict_ckpt.h"
#include "ckpt_data_handler/key_freq_map_ckpt/key_freq_map_ckpt.h"
#include "ckpt_data_handler/key_count_map_ckpt/key_count_map_ckpt.h"
#include "utils/time_cost.h"
#include "utils/common.h"
#include "file_system/file_system_handler.h"

#include "checkpoint.h"

using namespace std;
using namespace MxRec;

void Checkpoint::SaveModel(string savePath, CkptData& ckptData, RankInfo& mgmtRankInfo, const vector<EmbInfo>& embInfo)
{
    processPath = savePath;
    rankId = mgmtRankInfo.rankId;
    deviceId = mgmtRankInfo.deviceId;
    rankSize = mgmtRankInfo.rankSize;
    useDynamicExpansion = mgmtRankInfo.useDynamicExpansion;
    mgmtEmbInfo = embInfo;

    auto fileSystemHandler = make_unique<FileSystemHandler>();
    fileSystemPtr = fileSystemHandler->Create(savePath);

    LOG_INFO("Start host side saving data.");
    LOG_DEBUG("==Start to create save data handler.");
    SetDataHandler(ckptData);
    LOG_DEBUG("==Start save data process.");
    SaveProcess(ckptData);
    LOG_INFO("Finish host side saving data.");
}

void Checkpoint::LoadModel(string loadPath, CkptData& ckptData, RankInfo& mgmtRankInfo, const vector<EmbInfo>& embInfo,
                           const vector<CkptFeatureType>& featureTypes)
{
    processPath = loadPath;
    rankId = mgmtRankInfo.rankId;
    deviceId = mgmtRankInfo.deviceId;
    useDynamicExpansion = mgmtRankInfo.useDynamicExpansion;
    mgmtEmbInfo = embInfo;

    auto fileSystemHandler = make_unique<FileSystemHandler>();
    fileSystemPtr = fileSystemHandler->Create(loadPath);

    LOG_INFO("Start host side loading data.");
    LOG_DEBUG("==Start to create load data handler.");
    SetDataHandler(featureTypes);
    LOG_DEBUG("==Start load data process.");
    LoadProcess(ckptData);
    LOG_INFO("Finish host side loading data.");
}

void Checkpoint::SetDataHandler(CkptData& ckptData)
{
    dataHandlers.clear();
    if (!ckptData.keyCountMap.empty()) {
        dataHandlers.push_back(make_unique<KeyCountMapCkpt>());
    }
    if (!ckptData.table2Thresh.empty() && !ckptData.histRec.timestamps.empty() &&
        !ckptData.histRec.historyRecords.empty()) {
        dataHandlers.push_back(make_unique<FeatAdmitNEvictCkpt>());
    }
    if (!ckptData.ddrKeyFreqMaps.empty() && !ckptData.excludeDDRKeyFreqMaps.empty()) {
        dataHandlers.push_back(make_unique<KeyFreqMapCkpt>());
    }
}

void Checkpoint::SetDataHandler(const vector<CkptFeatureType>& featureTypes)
{
    map<CkptFeatureType, function<void()>> setCkptMap{
        {CkptFeatureType::FEAT_ADMIT_N_EVICT, [this] { dataHandlers.push_back(make_unique<FeatAdmitNEvictCkpt>()); }},
        {CkptFeatureType::DDR_KEY_FREQ_MAP,   [this] { dataHandlers.push_back(make_unique<KeyFreqMapCkpt>()); }},
        {CkptFeatureType::KEY_COUNT_MAP,      [this] { dataHandlers.push_back(make_unique<KeyCountMapCkpt>()); }}
    };

    for (const auto& featureType : featureTypes) {
        setCkptMap.at(featureType)();
    }
}

void Checkpoint::SaveProcess(CkptData& ckptData)
{
    for (const auto& dataHandler : dataHandlers) {
        dataHandler->SetProcessData(ckptData);
        vector<string> embNames { dataHandler->GetEmbNames() };
        vector<CkptDataType> saveDataTypes { dataHandler->GetDataTypes() };
        MakeUpperLayerSaveDir();
        MakeDataLayerSaveDir(embNames, saveDataTypes, dataHandler);
        SaveDataset(embNames, saveDataTypes, dataHandler);
    }
}

void Checkpoint::MakeUpperLayerSaveDir()
{
    innerDirPath = processPath;
    MakeSaveDir(innerDirPath);
}

void Checkpoint::MakeDataLayerSaveDir(const vector<string>& embNames,
                                      const vector<CkptDataType>& saveDataTypes,
                                      const unique_ptr<CkptDataHandler>& dataHandler)
{
    for (const auto& embName : embNames) {
        auto dataDir { innerDirPath + dirSeparator + embName };
        MakeSaveDir(dataDir);

        for (const auto& saveDataType : saveDataTypes) {
            auto dataDirName { dataHandler->GetDataDirName(saveDataType) };
            auto datasetPath { dataDir + dirSeparator + dataDirName };
            MakeSaveDir(datasetPath);
        }
    }
}

void Checkpoint::MakeSaveDir(const string& dirName) const
{
    if (fileSystemPtr == nullptr) {
        LOG_WARN("please init file system pointer before using. ");
        throw runtime_error("Nullptr. file system pointer is not initialized. ");
    }
    fileSystemPtr->CreateDir(dirName);
}

Checkpoint::EmbSizeInfo Checkpoint::GetEmbeddingSize(const string& embName)
{
    EmbSizeInfo embSizeInfo;
    for (const auto &embInfo: mgmtEmbInfo) {
        if (embInfo.name == embName) {
            embSizeInfo.embSize = embInfo.embeddingSize;
            embSizeInfo.extEmbSize = embInfo.extEmbeddingSize;
            return embSizeInfo;
        }
    }
    return embSizeInfo;
}

bool Checkpoint::CheckEmbNames(const string& embName)
{
    for (const auto &embInfo: mgmtEmbInfo) {
        if (embInfo.name == embName && embInfo.isSave)  {
            return true;
        }
    }
    return false;
}

void Checkpoint::SaveDataset(const vector<string>& embNames,
                             const vector<CkptDataType>& saveDataTypes,
                             const unique_ptr<CkptDataHandler>& dataHandler)
{
    for (const auto& embName: embNames) {
        if (!CheckEmbNames(embName)) {
            continue;
        }
        auto dataDir{innerDirPath + dirSeparator + embName};
        for (const auto& saveDataType: saveDataTypes) {
            auto datasetPath { dataDir + dirSeparator + dataHandler->GetDataDirName(saveDataType) };
            auto datasetDir { datasetPath + dirSeparator + datasetName + to_string(rankId) + dataFileType };

            LOG_DEBUG("====Start getting data from handler to: {}", datasetDir);
            auto transData { dataHandler->GetDataset(saveDataType, embName) };

            LOG_DEBUG("====Start saving data to: {}", datasetDir);
            WriteStream(transData, datasetDir, transData.datasetSize, saveDataType);
        }
    }
}

void Checkpoint::WriteStream(CkptTransData& transData, const string& dataDir, size_t dataSize, CkptDataType dataType)
{
    if (fileSystemPtr == nullptr) {
        LOG_WARN("please init file system pointer before using. ");
        throw runtime_error("Nullptr. file system pointer is not initialized. ");
    }

    ssize_t writeBytesNum;
    if (floatTransSet.find(dataType) != floatTransSet.end()) {
        writeBytesNum = fileSystemPtr->Write(dataDir, transData.floatArr, dataSize);
    } else if (int32TransSet.find(dataType) != int32TransSet.end()) {
        writeBytesNum = fileSystemPtr->Write(dataDir,
                                             reinterpret_cast<const char*>(transData.int32Arr.data()), dataSize);
    } else if (int64TransSet.find(dataType) != int64TransSet.end()) {
        writeBytesNum = fileSystemPtr->Write(dataDir,
                                             reinterpret_cast<const char*>(transData.int64Arr.data()), dataSize);
    } else if (dataType == CkptDataType::ATTRIBUTE) {
        writeBytesNum = fileSystemPtr->Write(dataDir,
                                             reinterpret_cast<const char*>(transData.attribute.data()), dataSize);
    }

    if (writeBytesNum == -1) {
        throw runtime_error(
                StringFormat("Error: Save data failed. data type: {} .An error occurred while writing file: {}.",
                             dataType, dataDir));
    }
    if (writeBytesNum != dataSize) {
        throw runtime_error(StringFormat(
                "Error: Save data failed. data type: {} .Expected to write {} bytes, but actually write {} bytes to file {}.",
                dataType, dataSize, writeBytesNum, dataDir));
    }
}


void Checkpoint::LoadProcess(CkptData& ckptData)
{
    for (const auto& dataHandler : dataHandlers) {
        vector<string> embNames {};
        vector<string> dirNames { dataHandler->GetDirNames() };
        vector<CkptDataType> saveDataTypes { dataHandler->GetDataTypes() };
        innerDirPath = processPath;
        if (find(dirNames.begin(), dirNames.end(), ssdSymbol) != dirNames.end()) {
            embNames = GetTableLayerLoadDir();
        } else {
            embNames = GetEmbedTableNames();
        }
        LoadDataset(embNames, saveDataTypes, dataHandler, ckptData);
        dataHandler->GetProcessData(ckptData);
    }
}


vector<string> Checkpoint::GetEmbedTableNames()
{
    vector<string> loadTableNames;
    for (const auto& embInfo : mgmtEmbInfo) {
        if (embInfo.isSave) {
            loadTableNames.push_back(embInfo.name);
        }
    }

    return loadTableNames;
}

vector<string> Checkpoint::GetTableLayerLoadDir()
{
    vector<string> loadTableDir;
    if (fileSystemPtr == nullptr) {
        LOG_WARN("please init file system pointer before using. ");
        throw runtime_error("Nullptr. file system pointer is not initialized. ");
    }
    loadTableDir = fileSystemPtr->ListDir(innerDirPath);
    return loadTableDir;
}

void Checkpoint::LoadDataset(const vector<string>& embNames,
                             const vector<CkptDataType>& saveDataTypes,
                             const unique_ptr<CkptDataHandler>& dataHandler,
                             CkptData& ckptData)
{
    for (const auto& embName : embNames) {
        auto dataDir { innerDirPath + dirSeparator + embName };
        for (const auto& saveDataType : saveDataTypes) {
            auto datasetPath { dataDir + dirSeparator + dataHandler->GetDataDirName(saveDataType) };

            auto datasetDir { datasetPath + dirSeparator + "slice" + dataFileType };
            auto attributeDir { datasetPath + dirSeparator + "slice" + attribFileType };

            CkptTransData transData;

            LOG_DEBUG("====Start reading data from: {}", attributeDir);
            auto dataElmtBytes { dataHandler->GetDataElmtBytes(CkptDataType::ATTRIBUTE) };
            ReadStream(transData, attributeDir, CkptDataType::ATTRIBUTE, dataElmtBytes);

            dataElmtBytes = dataHandler->GetDataElmtBytes(saveDataType);
            if (saveDataType == CkptDataType::EMB_DATA) {
                ReadStreamForEmbData(transData, datasetDir, dataElmtBytes, ckptData, embName);
                continue;
            } else {
                LOG_DEBUG("====Start reading data from: {}", datasetDir);
                ReadStream(transData, datasetDir, saveDataType, dataElmtBytes);
            }

            LOG_DEBUG("====Start loading data from: {} to data handler.", attributeDir);
            if ((saveDataType == CkptDataType::EMB_INFO))  {
                dataHandler->SetDatasetForLoadEmb(saveDataType, embName, transData, ckptData);
            } else {
                dataHandler->SetDataset(saveDataType, embName, transData);
            }
        }
    }
}

void Checkpoint::ReadStream(CkptTransData& transData,
                            const string& dataDir,
                            CkptDataType dataType,
                            uint32_t dataElmtBytes)
{
    if (dataElmtBytes == 0) {
        LOG_WARN("dataElmtBytes is 0, don't handle [/ %] operation");
        return ;
    }

    if (fileSystemPtr == nullptr) {
        LOG_WARN("please init file system pointer before using. ");
        throw runtime_error("Nullptr. file system pointer is not initialized. ");
    }

    size_t datasetSize = fileSystemPtr->GetFileSize(dataDir);
    auto resizeSize { datasetSize / dataElmtBytes };
    SetTransDataSize(transData, resizeSize, dataType);

    if (datasetSize % dataElmtBytes > 0) {
        LOG_DEBUG("data is missing or incomplete in load file: {}", dataDir);
    }

    ssize_t readBytesNum;
    if (int32TransSet.find(dataType) != int32TransSet.end()) {
        readBytesNum = fileSystemPtr->Read(dataDir, reinterpret_cast<char*>(transData.int32Arr.data()), datasetSize);
    } else if (int64TransSet.find(dataType) != int64TransSet.end()) {
        readBytesNum = fileSystemPtr->Read(dataDir, reinterpret_cast<char*>(transData.int64Arr.data()), datasetSize);
    } else if (floatTransSet.find(dataType) != floatTransSet.end()) {
        readBytesNum = fileSystemPtr->Read(dataDir, reinterpret_cast<char*>(transData.floatArr.data()), datasetSize);
    } else if (dataType == CkptDataType::ATTRIBUTE) {
        readBytesNum = fileSystemPtr->Read(dataDir, reinterpret_cast<char *>(transData.attribute.data()), datasetSize);
    }

    if (readBytesNum == -1) {
        throw runtime_error(
                StringFormat("Error: Load data failed. data type: {} .An error occurred while reading file: {}.",
                             dataType, dataDir));
    }
    if (readBytesNum != datasetSize) {
        throw runtime_error(StringFormat(
                "Error: Load data failed. data type: {} .Expected to read {} bytes, but actually read {} bytes to file {}.",
                dataType, datasetSize, readBytesNum, dataDir));
    }
}

void Checkpoint::ReadStreamForEmbData(CkptTransData& transData,
                                      const string& dataDir,
                                      uint32_t dataElmtBytes,
                                      CkptData& ckptData,
                                      string embName) const
{
    if (dataElmtBytes == 0) {
        LOG_ERROR("dataElmtBytes is 0, don't handle [/ %] operation");
        return ;
    }

    if (fileSystemPtr == nullptr) {
        LOG_WARN("please init file system pointer before using. ");
        throw runtime_error("Nullptr. file system pointer is not initialized. ");
    }

    auto embDataOuterSize = transData.attribute.at(attribEmbDataOuterIdx);
    if (embDataOuterSize <= 0 || embDataOuterSize > MAX_VOCABULARY_SIZE) {
        throw runtime_error(StringFormat("Invalid embDataOuterSize :%d", embDataOuterSize).c_str());
    }

    size_t datasetSize = fileSystemPtr->GetFileSize(dataDir);
    if (datasetSize % embDataOuterSize > 0 || datasetSize % dataElmtBytes > 0) {
        LOG_ERROR("data is missing or incomplete in load file: {}", dataDir);
        throw runtime_error("unable to load EMB_DATA cause wrong-format saved emb data");
    }

    auto loadHostEmbs = ckptData.hostEmbs;
    auto& dst = (*loadHostEmbs)[embName].embData;
    dst.reserve(embDataOuterSize);
}

void Checkpoint::SetTransDataSize(CkptTransData& transData, size_t datasetSize, CkptDataType dataType)
{
    if (int32TransSet.find(dataType) != int32TransSet.end()) {
        transData.int32Arr.resize(datasetSize);
    } else if (int64TransSet.find(dataType) != int64TransSet.end()) {
        transData.int64Arr.resize(datasetSize);
    } else if (floatTransSet.find(dataType) != floatTransSet.end()) {
        transData.floatArr.resize(datasetSize);
    } else if (dataType == CkptDataType::ATTRIBUTE) {
        transData.attribute.resize(datasetSize);
    }
}
