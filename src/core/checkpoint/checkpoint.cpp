/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-15
 */

#include <iostream>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/stat.h>

#include "ckpt_data_handler//emb_hash_ckpt/emb_hash_ckpt.h"
#include "ckpt_data_handler/host_emb_ckpt/host_emb_ckpt.h"
#include "ckpt_data_handler/nddr_offset_ckpt/nddr_offset_ckpt.h"
#include "ckpt_data_handler/nddr_feat_map_ckpt/nddr_feat_map_ckpt.h"
#include "ckpt_data_handler/feat_admit_n_evict_ckpt/feat_admit_n_evict_ckpt.h"
#include "utils/time_cost.h"
#include "utils/common.h"

#include "checkpoint.h"

using namespace std;
using namespace MxRec;

void Checkpoint::SaveModel(string savePath, CkptData& ckptData, RankInfo& mgmtRankInfo, const vector<EmbInfo>& EmbInfo)
{
    processPath = savePath;
    rankId = mgmtRankInfo.rankId;
    deviceId = mgmtRankInfo.deviceId;
    useDynamicExpansion = mgmtRankInfo.useDynamicExpansion;
    mgmtEmbInfo = EmbInfo;

    LOG(INFO) << "Start host side saving data.";
    VLOG(GLOG_DEBUG) << "==Start to create save data handler.";
    SetDataHandler(ckptData);
    VLOG(GLOG_DEBUG) << "==Start save data process.";
    SaveProcess(ckptData);
    LOG(INFO) << "Finish host side saving data.";
}

void Checkpoint::LoadModel(string loadPath, CkptData& ckptData, RankInfo& mgmtRankInfo, const vector<EmbInfo>& EmbInfo,
                           const vector<CkptFeatureType>& featureTypes)
{
    processPath = loadPath;
    rankId = mgmtRankInfo.rankId;
    deviceId = mgmtRankInfo.deviceId;
    useDynamicExpansion = mgmtRankInfo.useDynamicExpansion;
    mgmtEmbInfo = EmbInfo;

    LOG(INFO) << "Start host side loading data.";
    VLOG(GLOG_DEBUG) << "==Start to create load data handler.";
    SetDataHandler(featureTypes);
    VLOG(GLOG_DEBUG) << "==Start load data process.";
    LoadProcess(ckptData);
    LOG(INFO) << "Finish host side loading data.";
}

void Checkpoint::SetDataHandler(CkptData& ckptData)
{
    dataHandlers.clear();
    if (ckptData.hostEmbs != nullptr) {
        dataHandlers.push_back(make_unique<HostEmbCkpt>());
    }
    if (!ckptData.embHashMaps.empty()) {
        dataHandlers.push_back(make_unique<EmbHashCkpt>());
    }
    if (!ckptData.maxOffset.empty()) {
        dataHandlers.push_back(make_unique<NddrOffsetCkpt>());
    }
    if (!ckptData.keyOffsetMap.empty()) {
        dataHandlers.push_back(make_unique<NddrFeatMapCkpt>());
    }
    if (!ckptData.table2Thresh.empty() && !ckptData.histRec.timestamps.empty() &&
        !ckptData.histRec.historyRecords.empty()) {
        dataHandlers.push_back(make_unique<FeatAdmitNEvictCkpt>());
    }
}

void Checkpoint::SetDataHandler(const vector<CkptFeatureType>& featureTypes)
{
    map<CkptFeatureType, function<void()>> setCkptMap { { CkptFeatureType::HOST_EMB,
        [&] { dataHandlers.push_back(make_unique<HostEmbCkpt>()); } },
        { CkptFeatureType::EMB_HASHMAP, [&] { dataHandlers.push_back(make_unique<EmbHashCkpt>()); } },
        { CkptFeatureType::MAX_OFFSET, [&] { dataHandlers.push_back(make_unique<NddrOffsetCkpt>()); } },
        { CkptFeatureType::KEY_OFFSET_MAP, [&] { dataHandlers.push_back(make_unique<NddrFeatMapCkpt>()); } },
        { CkptFeatureType::FEAT_ADMIT_N_EVICT, [&] { dataHandlers.push_back(make_unique<FeatAdmitNEvictCkpt>()); } } };

    for (const auto& featureType : featureTypes) {
        setCkptMap.at(featureType)();
    }
}

void Checkpoint::SaveProcess(CkptData& ckptData)
{
    for (const auto& dataHandler : dataHandlers) {
        dataHandler->SetProcessData(ckptData);

        vector<string> embNames { dataHandler->GetEmbNames() };
        vector<string> dirNames { dataHandler->GetDirNames() };
        vector<CkptDataType> saveDataTypes { dataHandler->GetDataTypes() };

        MakeUpperLayerSaveDir(dirNames);
        MakeDataLayerSaveDir(embNames, saveDataTypes, dataHandler);

        SaveDataset(embNames, saveDataTypes, dataHandler);
    }
}

void Checkpoint::MakeUpperLayerSaveDir(const vector<string>& dirNames)
{
    innerDirPath = processPath;
    MakeSaveDir(innerDirPath);

    for (const auto& dirName : dirNames) {
        innerDirPath = innerDirPath + dirSeparator + dirName;
        MakeSaveDir(innerDirPath);
    }
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

void Checkpoint::MakeSaveDir(const string& dirName)
{
    if (access(dirName.c_str(), F_OK) == -1) {
        if (mkdir(dirName.c_str(), dirMode) == -1) {
            VLOG(GLOG_DEBUG) << StringFormat("Unable to create directory: %s", dirName.c_str());
        }
    }
}

int Checkpoint::GetEmbeddingSize(const string& embName) const
{
    for (const auto &embInfo: mgmtEmbInfo) {
        if (embInfo.name == embName) {
            return embInfo.extEmbeddingSize;
        }
    }
    return 0;
}

bool Checkpoint::CheckEmbNames(const string& embName)
{
    for (const auto &embInfo: mgmtEmbInfo) {
        if (embInfo.name == embName && embInfo.isSave == true)  {
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
            auto attributeDir { datasetPath + dirSeparator + datasetName + to_string(rankId) + attribFileType };

            VLOG(GLOG_DEBUG) << StringFormat("====Start getting data from handler to: %s", datasetDir.c_str());
            auto transData { dataHandler->GetDataset(saveDataType, embName) };

            // save embedding when dynamic expansion is open
            if ((saveDataType == CkptDataType::NDDR_FEATMAP) && useDynamicExpansion) {
                auto embedPath { dataDir + dirSeparator + "key_embedding" };
                auto embedDatasetDir { embedPath + dirSeparator + datasetName + to_string(rankId) + dataFileType };
                auto embeddingSize = GetEmbeddingSize(embName);
                MakeSaveDir(embedPath);
                VLOG(GLOG_DEBUG) << StringFormat("====Start saving embedding data to: %s", datasetDir.c_str());
                WriteEmbedding(transData, embedDatasetDir, embeddingSize);
            }

            VLOG(GLOG_DEBUG) << StringFormat("====Start saving data to: %s", datasetDir.c_str());
            WriteStream(transData, datasetDir, transData.datasetSize, saveDataType);
            VLOG(GLOG_DEBUG) << StringFormat("====Start saving data to: %s", attributeDir.c_str());
            WriteStream(transData, attributeDir, transData.attributeSize, CkptDataType::ATTRIBUTE);
        }
    }
}

void Checkpoint::WriteEmbedding(const CkptTransData& transData, const string& dataDir, const int& embeddingSize)
{
    ofstream writeFile;
    writeFile.open(dataDir.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

#ifndef GTEST
    auto res = aclrtSetDevice(static_cast<int32_t>(deviceId));
    if (res != ACL_ERROR_NONE) {
        LOG(ERROR) << StringFormat("Set device failed, device_id:%d", deviceId);
        throw runtime_error(StringFormat("Set device failed, device_id:%d", deviceId).c_str());
    }

    auto &transArr = transData.int64Arr;
    for (size_t i{0}; i < transArr.size(); i += embHashNum) {
        vector<float> row(embeddingSize);
        int64_t address = transArr.at(i + 1);
        float *floatPtr = reinterpret_cast<float *>(address);

        aclError ret = aclrtMemcpy(row.data(), embeddingSize * sizeof(float),
                                   floatPtr, embeddingSize * sizeof(float),
                                   ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG(ERROR) << StringFormat("aclrtMemcpy failed, ret=%d", ret);
            throw runtime_error(StringFormat("aclrtMemcpy failed, ret=%d", ret).c_str());
        }

        writeFile.write((const char *) (row.data()), embeddingSize * sizeof(float));
    }
#endif
    writeFile.close();
}

void Checkpoint::ReadEmbedding(CkptTransData& transData, const string& dataDir)
{
    std::ifstream readFile;
    readFile.open(dataDir.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    size_t datasetSize = static_cast<size_t>(readFile.tellg());
    readFile.seekg(0, std::ios::beg);
    try {
        ValidateReadFile(dataDir, datasetSize);
    } catch (const std::invalid_argument& e) {
        readFile.close();
        throw runtime_error(StringFormat("Invalid read file path: %s", e.what()));
    }

#ifndef GTEST
    auto res = aclrtSetDevice(static_cast<int32_t>(deviceId));
    if (res != ACL_ERROR_NONE) {
        LOG(ERROR) << StringFormat("Set device failed, device_id:%d", deviceId);
        readFile.close();
        throw runtime_error(StringFormat("Set device failed, device_id:%d", deviceId).c_str());
    }

    auto &AttributeArr = transData.attribute;
    auto embHashMapSize = AttributeArr.at(0);
    if (embHashMapSize <= 0) {
        throw runtime_error(StringFormat("Invalid EmbHashMapSize:%d, must be greater than 0", embHashMapSize).c_str());
    }
    auto embeddingSize = static_cast<int>(datasetSize / sizeof(float) / embHashMapSize);

    aclError ret;
    void *newBlock = nullptr;
    ret = aclrtMalloc(&newBlock, static_cast<int>(datasetSize), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG(ERROR) << StringFormat("aclrtMalloc failed, ret=%d", ret);
        readFile.close();
        throw runtime_error(StringFormat("aclrtMemcpy failed, ret=%d", ret).c_str());
    }

    float *floatPtr = static_cast<float *>(newBlock);
    auto &transArr = transData.int64Arr;
    for (size_t i{0}; i < transArr.size(); i += embHashNum) {
        vector<float> row(embeddingSize);
        readFile.read((char *) (row.data()), embeddingSize * sizeof(float));

        aclError ret = aclrtMemcpy(floatPtr + i * embeddingSize, embeddingSize * sizeof(float),
                                   row.data(), embeddingSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            LOG(ERROR) << StringFormat("aclrtMemcpy failed, ret=%d", ret);
            readFile.close();
            throw runtime_error(StringFormat("aclrtMemcpy failed, ret=%d", ret).c_str());
        }
        int64_t address = reinterpret_cast<int64_t>(floatPtr + i * embeddingSize);
        transArr.at(i + 1) = address;
    }
#endif
    readFile.close();
}

void Checkpoint::WriteStream(CkptTransData& transData, const string& dataDir, size_t dataSize, CkptDataType dataType)
{
    ofstream writeFile;
    writeFile.open(dataDir.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

    if (!writeFile.is_open()) {
        VLOG(GLOG_DEBUG) << StringFormat("unable to open save file: %s", dataDir.c_str());
        writeFile.close();
        return;
    }

    int loops = 1;
    if (dataType == CkptDataType::EMB_DATA) {
        loops = static_cast<int>(transData.floatArr.size());
    }
    for (int i = 0; i < loops; i++) {
        size_t idx = 0;
        size_t writeSize = 0;
        size_t dataCol = dataSize;
        while (dataCol != 0) {
            if (dataCol > oneTimeReadWriteLen) {
                writeSize = oneTimeReadWriteLen;
            } else {
                writeSize = dataCol;
            }
            if (floatTransSet.find(dataType) != floatTransSet.end()) {
                writeFile.write((const char*)(transData.floatArr[i]) + idx, writeSize);
            } else {
                WriteDataset(transData, writeFile, writeSize, dataType, idx);
            }

            dataCol -= writeSize;
            idx += writeSize;
        }
    }

    writeFile.close();
}

void Checkpoint::WriteDataset(CkptTransData& transData,
                              ofstream& writeFile,
                              size_t writeSize,
                              CkptDataType dataType,
                              size_t idx)
{
    if (int32TransSet.find(dataType) != int32TransSet.end()) {
        writeFile.write((const char*)(transData.int32Arr.data()) + idx, writeSize);
    } else if (int64TransSet.find(dataType) != int64TransSet.end()) {
        writeFile.write((const char*)(transData.int64Arr.data()) + idx, writeSize);
    } else if (dataType == CkptDataType::ATTRIBUTE) {
        writeFile.write((const char*)(transData.attribute.data()) + idx, writeSize);
    }
}

void Checkpoint::LoadProcess(CkptData& ckptData)
{
    for (const auto& dataHandler : dataHandlers) {
        vector<string> embNames {};
        vector<string> dirNames { dataHandler->GetDirNames() };
        vector<CkptDataType> saveDataTypes { dataHandler->GetDataTypes() };

        GetUpperLayerLoadDir(dirNames);
        embNames = GetEmbedTableNames();

        LoadDataset(embNames, saveDataTypes, dataHandler, ckptData);

        dataHandler->GetProcessData(ckptData);
    }
}

void Checkpoint::GetUpperLayerLoadDir(const vector<string>& dirNames)
{
    innerDirPath = processPath;

    for (const auto& dirName : dirNames) {
        innerDirPath = innerDirPath + dirSeparator + dirName;
    }
}

vector<string> Checkpoint::GetEmbedTableNames()
{
    vector<string> loadTableNames;
    for (const auto& embInfo : mgmtEmbInfo) {
        if (embInfo.isSave == true) {
            loadTableNames.push_back(embInfo.name);
        }
    }

    return loadTableNames;
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

            auto datasetDir { datasetPath + dirSeparator + datasetName + to_string(rankId) + dataFileType };
            auto attributeDir { datasetPath + dirSeparator + datasetName + to_string(rankId) + attribFileType };

            CkptTransData transData;

            VLOG(GLOG_DEBUG) << StringFormat("====Start reading data from: %s", attributeDir.c_str());
            auto dataElmtBytes { dataHandler->GetDataElmtBytes(CkptDataType::ATTRIBUTE) };
            ReadStream(transData, attributeDir, CkptDataType::ATTRIBUTE, dataElmtBytes);

            dataElmtBytes = dataHandler->GetDataElmtBytes(saveDataType);
            if (saveDataType == CkptDataType::EMB_DATA) {
                ReadStreamForEmbData(transData, datasetDir, dataElmtBytes, ckptData, embName);
                continue;
            } else {
                VLOG(GLOG_DEBUG) << StringFormat("====Start reading data from: %s", datasetDir.c_str());
                ReadStream(transData, datasetDir, saveDataType, dataElmtBytes);
            }

            // load embedding when use dynamic expansion is open
            if ((saveDataType == CkptDataType::NDDR_FEATMAP) && useDynamicExpansion)  {
                auto embedPath { dataDir + dirSeparator + "key_embedding" };
                auto embedDatasetDir { embedPath + dirSeparator + datasetName + to_string(rankId) + dataFileType };
                VLOG(GLOG_DEBUG) << StringFormat("====Start loading embedding data from: %s", datasetDir.c_str());
                ReadEmbedding(transData, embedDatasetDir);
            }

            VLOG(GLOG_DEBUG) << StringFormat(
                "====Start loading data from: %s to data handler.", attributeDir.c_str()
            );
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
        LOG(WARNING) << "dataElmtBytes is 0, don't handle [/ %] operation";
        return ;
    }

    std::ifstream readFile;
    readFile.open(dataDir.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    size_t datasetSize = static_cast<size_t>(readFile.tellg());
    readFile.seekg(0, std::ios::beg);
    try {
        ValidateReadFile(dataDir, datasetSize);
    } catch (const std::invalid_argument& e) {
        readFile.close();
        throw runtime_error(StringFormat("Invalid read file path: %s", e.what()));
    }

    if (datasetSize % dataElmtBytes > 0) {
        VLOG(GLOG_DEBUG) << StringFormat("data is missing or incomplete in load file: %s", dataDir.c_str());
    }

    auto resizeSize { datasetSize / dataElmtBytes };
    SetTransDataSize(transData, resizeSize, dataType);

    if (readFile.is_open()) {
        size_t idx = 0;
        size_t readSize = 0;
        while (datasetSize != 0) {
            if (datasetSize > oneTimeReadWriteLen) {
                readSize = oneTimeReadWriteLen;
            } else {
                readSize = datasetSize;
            }
            ReadDataset(transData, readFile, readSize, dataType, idx);
            datasetSize -= readSize;
            idx += readSize;
        }
    } else {
        VLOG(GLOG_DEBUG) << StringFormat("unable to open load file: %s", dataDir.c_str());
    }

    readFile.close();
}

void Checkpoint::ReadStreamForEmbData(CkptTransData& transData,
                                      const string& dataDir,
                                      uint32_t dataElmtBytes,
                                      CkptData& ckptData,
                                      string embName)
{
    if (dataElmtBytes == 0) {
        LOG(ERROR) << "dataElmtBytes is 0, don't handle [/ %] operation";
        return ;
    }
    auto embDataOuterSize = transData.attribute.at(attribEmbDataOuterIdx);
    if (embDataOuterSize <= 0 || embDataOuterSize > MAX_VOCABULARY_SIZE) {
        throw runtime_error(StringFormat("Invalid embDataOuterSize :%d", embDataOuterSize).c_str());
    }
    std::ifstream readFile;
    readFile.open(dataDir.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    size_t datasetSize = static_cast<size_t>(readFile.tellg());
    readFile.seekg(0, std::ios::beg);
    try {
        ValidateReadFile(dataDir, datasetSize);
    } catch (const std::invalid_argument& e) {
        readFile.close();
        throw runtime_error(StringFormat("Invalid read file path: %s", e.what()));
    }

    if (datasetSize % embDataOuterSize > 0 || datasetSize % dataElmtBytes > 0) {
        LOG(ERROR) << StringFormat("data is missing or incomplete in load file: %s", dataDir.c_str());
        readFile.close();
        throw runtime_error("unable to load EMB_DATA cause wrong-format saved emb data");
    }
    auto loadHostEmbs = ckptData.hostEmbs;
    auto& dst = (*loadHostEmbs)[embName].embData;
    dst.reserve(embDataOuterSize);
    auto onceReadByteSize { datasetSize / embDataOuterSize };

    if (!readFile.is_open()) {
        VLOG(GLOG_DEBUG) << StringFormat("unable to open load file: %s", dataDir.c_str());
        readFile.close();
        return;
    }
    for (size_t i = 0; i < embDataOuterSize; ++i) {
        size_t idx = 0;
        size_t readSize = 0;
        size_t dataCol = onceReadByteSize;
        while (dataCol != 0) {
            if (dataCol > oneTimeReadWriteLen) {
                readSize = oneTimeReadWriteLen;
            } else {
                readSize = dataCol;
            }
            readFile.read((char*)(dst[i].data()) + idx, readSize);
            dataCol -= readSize;
            idx += readSize;
        }
    }
    readFile.close();
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

void Checkpoint::ReadDataset(CkptTransData& transData,
                             ifstream& readFile,
                             size_t readSize,
                             CkptDataType dataType,
                             size_t idx)
{
    if (int32TransSet.find(dataType) != int32TransSet.end()) {
        readFile.read((char*)(transData.int32Arr.data()) + idx, readSize);
    } else if (int64TransSet.find(dataType) != int64TransSet.end()) {
        readFile.read((char*)(transData.int64Arr.data()) + idx, readSize);
    } else if (floatTransSet.find(dataType) != floatTransSet.end()) {
        readFile.read((char*)(transData.floatArr.data()) + idx, readSize);
    } else if (dataType == CkptDataType::ATTRIBUTE) {
        readFile.read((char*)(transData.attribute.data()) + idx, readSize);
    }
}
