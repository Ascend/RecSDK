/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-15
 */

#include <spdlog/spdlog.h>
#include <iostream>
#include <sys/mman.h>
#include <cstring>

#include "ckpt_data_handler//emb_hash_ckpt/emb_hash_ckpt.h"
#include "ckpt_data_handler/host_emb_ckpt/host_emb_ckpt.h"
#include "ckpt_data_handler/nddr_offset_ckpt/nddr_offset_ckpt.h"
#include "ckpt_data_handler/nddr_feat_map_ckpt/nddr_feat_map_ckpt.h"
#include "ckpt_data_handler/feat_admit_n_evict_ckpt/feat_admit_n_evict_ckpt.h"

#include "checkpoint.h"

using namespace std;
using namespace MxRec;

void Checkpoint::SaveModel(string savePath, CkptData& ckptData, RankInfo& mgmtRankInfo, const vector<EmbInfo>& EmbInfo)
{
    // TODO: check savePath

    processPath = savePath;
    rankId = mgmtRankInfo.rankId;
    deviceId = mgmtRankInfo.deviceId;
    useDynamicExpansion = mgmtRankInfo.useDynamicExpansion;
    mgmtEmbInfo = EmbInfo;

    spdlog::info("Start host side saving data.");
    spdlog::debug("==Start to create save data handler.");
    SetDataHandler(ckptData);
    spdlog::debug("==Start save data process.");
    SaveProcess(ckptData);
    spdlog::info("Finish host side saving data.");
}

void Checkpoint::LoadModel(string loadPath, CkptData& ckptData, RankInfo& mgmtRankInfo, const vector<EmbInfo>& EmbInfo,
                           const vector<CkptFeatureType>& featureTypes)
{
    // TODO: check loadPath

    processPath = loadPath;
    rankId = mgmtRankInfo.rankId;
    deviceId = mgmtRankInfo.deviceId;
    useDynamicExpansion = mgmtRankInfo.useDynamicExpansion;
    mgmtEmbInfo = EmbInfo;

    spdlog::info("Start host side loading data.");
    spdlog::debug("==Start to create load data handler.");
    SetDataHandler(featureTypes);
    spdlog::debug("==Start load data process.");
    LoadProcess(ckptData);
    spdlog::info("Finish host side loading data.");
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
    if (!ckptData.tens2Thresh.empty() && !ckptData.histRec.timestamps.empty() &&
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
    if (mkdir(dirName.c_str(), dirMode) == -1) {
        spdlog::debug("Unable to create directory: {}", dirName);
    }
}

int Checkpoint::GetEmbeddingSize(const string& embName)
{
    for (const auto &embInfo: mgmtEmbInfo) {
        if (embInfo.name == embName) {
            return embInfo.extEmbeddingSize;
        }
    }
    return 0;
}


void Checkpoint::SaveDataset(const vector<string>& embNames,
                             const vector<CkptDataType>& saveDataTypes,
                             const unique_ptr<CkptDataHandler>& dataHandler)
{
    for (const auto& embName: embNames) {
        auto dataDir{innerDirPath + dirSeparator + embName};
        for (const auto& saveDataType: saveDataTypes) {
            auto datasetPath { dataDir + dirSeparator + dataHandler->GetDataDirName(saveDataType) };
            auto datasetDir { datasetPath + dirSeparator + datasetName + to_string(rankId) + dataFileType };
            auto attributeDir { datasetPath + dirSeparator + datasetName + to_string(rankId) + attribFileType };

            spdlog::debug("====Start getting data from handler to: {}", datasetDir);
            auto transData { dataHandler->GetDataset(saveDataType, embName) };

            // save embedding when dynamic expansion is open
            if ((saveDataType == CkptDataType::NDDR_FEATMAP) && useDynamicExpansion) {
                auto embedPath { dataDir + dirSeparator + "key_embedding" };
                auto embedDatasetDir { embedPath + dirSeparator + datasetName + to_string(rankId) + dataFileType };
                auto embeddingSize = GetEmbeddingSize(embName);
                MakeSaveDir(embedPath);
                spdlog::debug("====Start saving embedding data to: {}", datasetDir);
                WriteEmbedding(transData, embedDatasetDir, embeddingSize);
            }

            spdlog::debug("====Start saving data to: {}", datasetDir);
            WriteStream(transData, datasetDir, transData.datasetSize, saveDataType);
            spdlog::debug("====Start saving data to: {}", attributeDir);
            WriteStream(transData, attributeDir, transData.attributeSize, CkptDataType::ATTRIBUTE);
        }
    }
}

void Checkpoint::WriteEmbedding(CkptTransData& transData, const string& dataDir, int& embeddingSize)
{
    ofstream writeFile;
    writeFile.open(dataDir.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);

#ifndef GTEST
    auto res = aclrtSetDevice(static_cast<int32_t>(deviceId));
    if (res != ACL_ERROR_NONE) {
        spdlog::error("Set device failed, device_id:{}", deviceId);
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
            spdlog::error("aclrtMemcpy failed, ret={}", ret);
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

#ifndef GTEST
    auto res = aclrtSetDevice(static_cast<int32_t>(deviceId));
    if (res != ACL_ERROR_NONE) {
        spdlog::error("Set device failed, device_id:{}", deviceId);
    }

    auto &AttributeArr = transData.attribute;
    auto embHashMapSize = AttributeArr.at(0);
    size_t datasetSize = readFile.tellg();
    readFile.seekg(0, std::ios::beg);
    auto embeddingSize = static_cast<int>(datasetSize / sizeof(float) / embHashMapSize);

    aclError ret;
    void *newBlock = nullptr;
    ret = aclrtMalloc(&newBlock, static_cast<int>(datasetSize), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        spdlog::error("aclrtMalloc failed, ret={}", ret);
    }

    float *floatPtr = static_cast<float *>(newBlock);
    auto &transArr = transData.int64Arr;
    for (size_t i{0}; i < transArr.size(); i += embHashNum) {
        vector<float> row(embeddingSize);
        readFile.read((char *) (row.data()), embeddingSize * sizeof(float));

        aclError ret = aclrtMemcpy(floatPtr + i * embeddingSize, embeddingSize * sizeof(float),
                                   row.data(), embeddingSize * sizeof(float),
                                   ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            spdlog::error("aclrtMemcpy failed, ret={}", ret);
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
        spdlog::debug("unable to open save file: {}", dataDir);
        writeFile.close();
        return;
    }

    int loops = 1;
    if (dataType == CkptDataType::EMB_DATA) {
        loops = transData.floatArr.size();
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
        embNames = GetTableLayerLoadDir();

        LoadDataset(embNames, saveDataTypes, dataHandler, ckptData);

        dataHandler->GetProcessData(ckptData);
    }
}

void Checkpoint::GetUpperLayerLoadDir(const vector<string>& dirNames)
{
    innerDirPath = processPath;
    // TODO: check existence

    for (const auto& dirName : dirNames) {
        innerDirPath = innerDirPath + dirSeparator + dirName;
        // TODO: check existence
    }
}

vector<string> Checkpoint::GetTableLayerLoadDir()
{
    vector<string> loadTableDir;
    auto dir { opendir(innerDirPath.c_str()) };
    struct dirent* en;
    if (dir != nullptr) {
        while ((en = readdir(dir)) != nullptr) {
            if (strcmp(en->d_name, currDir.c_str()) != 0 &&
                strcmp(en->d_name, prevDir.c_str()) != 0) {
                loadTableDir.emplace_back(en->d_name);
            }
        }
        closedir(dir);
    }
    // TODO: may cause memory problem? need to check

    return loadTableDir;
}

void Checkpoint::LoadDataset(const vector<string>& embNames,
                             const vector<CkptDataType>& saveDataTypes,
                             const unique_ptr<CkptDataHandler>& dataHandler,
                             CkptData& ckptData)
{
    for (const auto& embName : embNames) {
        auto dataDir { innerDirPath + dirSeparator + embName };
        // TODO: check existence
        for (const auto& saveDataType : saveDataTypes) {
            auto datasetPath { dataDir + dirSeparator + dataHandler->GetDataDirName(saveDataType) };
            // TODO: check existence

            auto datasetDir { datasetPath + dirSeparator + datasetName + to_string(rankId) + dataFileType };
            auto attributeDir { datasetPath + dirSeparator + datasetName + to_string(rankId) + attribFileType };

            CkptTransData transData;

            spdlog::debug("====Start reading data from: {}", attributeDir);
            auto dataElmtBytes { dataHandler->GetDataElmtBytes(CkptDataType::ATTRIBUTE) };
            ReadStream(transData, attributeDir, CkptDataType::ATTRIBUTE, dataElmtBytes);

            dataElmtBytes = dataHandler->GetDataElmtBytes(saveDataType);
            if (saveDataType == CkptDataType::EMB_DATA) {
                ReadStreamForEmbData(transData, datasetDir, dataElmtBytes, ckptData, embName);
                continue;
            } else {
                spdlog::debug("====Start reading data from: {}", datasetDir);
                ReadStream(transData, datasetDir, saveDataType, dataElmtBytes);
            }

            // load embedding when use dynamic expansion is open
            if ((saveDataType == CkptDataType::NDDR_FEATMAP) && useDynamicExpansion)  {
                auto embedPath { dataDir + dirSeparator + "key_embedding" };
                auto embedDatasetDir { embedPath + dirSeparator + datasetName + to_string(rankId) + dataFileType };
                spdlog::debug("====Start loading embedding data from: {}", datasetDir);
                ReadEmbedding(transData, embedDatasetDir);
            }

            spdlog::debug("====Start loading data from: {} to data handler.", attributeDir);
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
        spdlog::error("dataElmtBytes is 0, don't handle [/ %] operation");
        return ;
    }
    std::ifstream readFile;
    readFile.open(dataDir.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

    size_t datasetSize = readFile.tellg();
    readFile.seekg(0, std::ios::beg);

    if (datasetSize % dataElmtBytes > 0) {
        spdlog::debug("data is missing or incomplete in load file: {}", dataDir);
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
        spdlog::debug("unable to open load file: {}", dataDir);
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
        spdlog::error("dataElmtBytes is 0, don't handle [/ %] operation");
        return ;
    }

    auto embDataOuterSize = transData.attribute.at(attribEmbDataOuterIdx);

    auto loadHostEmbs = ckptData.hostEmbs;
    auto& dst = (*loadHostEmbs)[embName].embData;
    dst.reserve(embDataOuterSize);

    std::ifstream readFile;
    readFile.open(dataDir.c_str(), std::ios::in | std::ios::binary | std::ios::ate);

    size_t datasetSize = readFile.tellg();
    readFile.seekg(0, std::ios::beg);

    if (datasetSize % embDataOuterSize > 0 || datasetSize % dataElmtBytes > 0) {
        spdlog::error("data is missing or incomplete in load file: {}", dataDir);
        throw runtime_error("unable to load EMB_DATA cause wrong-format saved emb data");
    }
    auto onceReadByteSize { datasetSize / embDataOuterSize };

    if (!readFile.is_open()) {
        spdlog::debug("unable to open load file: {}", dataDir);
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