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
#include <fcntl.h>
#include <omp.h>

#include "ckpt_data_handler//emb_hash_ckpt/emb_hash_ckpt.h"
#include "ckpt_data_handler/host_emb_ckpt/host_emb_ckpt.h"
#include "ckpt_data_handler/nddr_offset_ckpt/nddr_offset_ckpt.h"
#include "ckpt_data_handler/nddr_feat_map_ckpt/nddr_feat_map_ckpt.h"
#include "ckpt_data_handler/feat_admit_n_evict_ckpt/feat_admit_n_evict_ckpt.h"
#include "ckpt_data_handler/key_freq_map_ckpt/key_freq_map_ckpt.h"
#include "ckpt_data_handler/key_count_map_ckpt/key_count_map_ckpt.h"
#include "utils/time_cost.h"
#include "utils/common.h"
#include "checkpoint.h"

using namespace std;
using namespace MxRec;

void Checkpoint::SaveModel(string savePath, CkptData& ckptData, RankInfo& mgmtRankInfo, const vector<EmbInfo>& embInfo)
{
    processPath = savePath;
    rankId = mgmtRankInfo.rankId;
    deviceId = mgmtRankInfo.deviceId;
    useDynamicExpansion = mgmtRankInfo.useDynamicExpansion;
    mgmtEmbInfo = embInfo;

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
    if (!ckptData.ddrKeyFreqMaps.empty() && !ckptData.excludeDDRKeyFreqMaps.empty()) {
        dataHandlers.push_back(make_unique<KeyFreqMapCkpt>());
    }
}

void Checkpoint::SetDataHandler(const vector<CkptFeatureType>& featureTypes)
{
    map<CkptFeatureType, function<void()>> setCkptMap{
        {CkptFeatureType::HOST_EMB,           [this] { dataHandlers.push_back(make_unique<HostEmbCkpt>()); }},
        {CkptFeatureType::EMB_HASHMAP,        [this] { dataHandlers.push_back(make_unique<EmbHashCkpt>()); }},
        {CkptFeatureType::MAX_OFFSET,         [this] { dataHandlers.push_back(make_unique<NddrOffsetCkpt>()); }},
        {CkptFeatureType::KEY_OFFSET_MAP,     [this] { dataHandlers.push_back(make_unique<NddrFeatMapCkpt>()); }},
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

void Checkpoint::MakeSaveDir(const string& dirName) const
{
    if (access(dirName.c_str(), F_OK) == -1) {
        if (mkdir(dirName.c_str(), dirMode) == -1) {
            LOG_DEBUG("Unable to create directory: {}", dirName);
        }
    }
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
            auto attributeDir { datasetPath + dirSeparator + datasetName + to_string(rankId) + attribFileType };

            LOG_DEBUG("====Start getting data from handler to: {}", datasetDir);
            auto transData { dataHandler->GetDataset(saveDataType, embName) };

            // save embedding when dynamic expansion is open
            if ((saveDataType == CkptDataType::NDDR_FEATMAP) && useDynamicExpansion) {
                auto embedPath { dataDir + dirSeparator + "key_embedding" };
                auto embedDatasetDir { embedPath + dirSeparator + datasetName + to_string(rankId) + dataFileType };
                auto embeddingSizeInfo = GetEmbeddingSize(embName);
                MakeSaveDir(embedPath);
                LOG_DEBUG("====Start saving embedding data to: {}", datasetDir);
                WriteEmbedding(transData, embedDatasetDir, embeddingSizeInfo.extEmbSize);
            }

            LOG_DEBUG("====Start saving data to: {}", datasetDir);
            WriteStream(transData, datasetDir, transData.datasetSize, saveDataType);
            LOG_DEBUG("====Start saving data to: {}", attributeDir);
            WriteStream(transData, attributeDir, transData.attributeSize, CkptDataType::ATTRIBUTE);
        }
    }
}

void Checkpoint::WriteEmbedding(const CkptTransData& transData, const string& dataDir, const int& embeddingSize)
{
    ofstream writeFile;
    writeFile.open(dataDir.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
    fs::permissions(dataDir.c_str(), fs::perms::owner_read | fs::perms::owner_write | fs::perms::group_read);

#ifndef GTEST
    auto res = aclrtSetDevice(static_cast<int32_t>(deviceId));
    if (res != ACL_ERROR_NONE) {
        LOG_ERROR("Set device failed, device_id:{}", deviceId);
        writeFile.close();
        throw runtime_error(Logger::Format("Set device failed, device_id:{}", deviceId).c_str());
    }

    auto &transArr = transData.int64Arr;
    for (size_t i{0}; i < transArr.size(); i += embHashNum) {
        vector<float> row(embeddingSize);
        int64_t address = transArr.at(i + 1);
        float *floatPtr = reinterpret_cast<float *>(address);

        aclError ret;
        try {
            ret = aclrtMemcpy(row.data(), embeddingSize * sizeof(float),
                              floatPtr, embeddingSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        } catch (std::exception& e) {
            writeFile.close();
            throw runtime_error(StringFormat("error happen when acl memory copy from device to host: %s", e.what()));
        }

        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtMemcpy failed, ret={}", ret);
            writeFile.close();
            throw runtime_error(Logger::Format("aclrtMemcpy failed, ret={}", ret).c_str());
        }

        try {
            writeFile.write(reinterpret_cast<const char *>(row.data()), embeddingSize * sizeof(float));
        } catch (std::exception& e) {
            writeFile.close();
            throw runtime_error(StringFormat("error happen when write embedding to file: %s", e.what()));
        }
    }
#endif
    writeFile.close();
}

void Checkpoint::ReadEmbedding(CkptTransData& transData, const string& dataDir, const string& embName)
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
        LOG_ERROR("Set device failed, device_id:{}", deviceId);
        readFile.close();
        throw runtime_error(StringFormat("Set device failed, device_id:%d", deviceId).c_str());
    }

    auto &attributeArr = transData.attribute;
    auto embHashMapSize = attributeArr.at(0);
    if (embHashMapSize <= 0) {
        readFile.close();
        throw runtime_error(StringFormat("Invalid EmbHashMapSize:%d, must be greater than 0", embHashMapSize).c_str());
    }
    auto embeddingSize = static_cast<int>(datasetSize / sizeof(float) / embHashMapSize);
    EmbSizeInfo embSizeInfo = GetEmbeddingSize(embName);
    if (embeddingSize != embSizeInfo.extEmbSize) {
        readFile.close();
        throw runtime_error(StringFormat("Invalid embedding size to be read, may read file has been changed").c_str());
    }

    aclError ret;
    void *newBlock = nullptr;
    ret = aclrtMalloc(&newBlock, static_cast<int>(datasetSize), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMalloc failed, ret={}", ret);
        readFile.close();
        throw runtime_error(StringFormat("aclrtMemcpy failed, ret=%d", ret).c_str());
    }

    float *floatPtr = static_cast<float *>(newBlock);
    auto &transArr = transData.int64Arr;
    for (size_t i = 0, j = 0; i < transArr.size(); i += keyAddrElem, ++j) {
        vector<float> row(embeddingSize);
        try {
            readFile.read(reinterpret_cast<char *>(row.data()), embeddingSize * sizeof(float));
        } catch (std::exception& e) {
            readFile.close();
            throw runtime_error(StringFormat("error happen when reading embedding from file: %s", e.what()));
        }
        aclError ec;
        try {
            ec = aclrtMemcpy(floatPtr + j * embeddingSize, embeddingSize * sizeof(float),
                             row.data(), embeddingSize * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        } catch (std::exception& e) {
            readFile.close();
            throw runtime_error(StringFormat("error happen when acl memory copy from host to device: %s", e.what()));
        }
        if (ec != ACL_SUCCESS) {
            LOG_ERROR("aclrtMemcpy failed, ret={}", ec);
            readFile.close();
            throw runtime_error(StringFormat("aclrtMemcpy failed, ret=%d", ec).c_str());
        }
        int64_t address = reinterpret_cast<int64_t>(floatPtr + j * embeddingSize);
        transArr.at(i + 1) = address;
    }
#endif
    readFile.close();
}

void Checkpoint::WriteStream(CkptTransData& transData, const string& dataDir, size_t dataSize, CkptDataType dataType)
{
    int fd = open(dataDir.c_str(), O_RDWR | O_CREAT | O_TRUNC, static_cast<mode_t>(0640));
    if (fd == -1) {
        LOG_ERROR("Error opening file for writing");
        return;
    }

    buffer.reserve(BUFFER_SIZE);

    BufferQueue queue;

    std::thread writer(&Checkpoint::WriterFn, this, std::ref(queue), fd);

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
                FillToBuffer(queue, reinterpret_cast<const char*>(transData.floatArr[i]) + idx, writeSize);
            } else {
                WriteDataset(transData, fd, writeSize, dataType, idx);
            }

            dataCol -= writeSize;
            idx += writeSize;
        }
    }

    // After all data has been processed, check if there is any data left in the buffer
    if (!buffer.empty()) {
        queue.Push(std::move(buffer));
        buffer.clear();
    }

    queue.Push(std::vector<char>());

    writer.join();

    close(fd);
}

void Checkpoint::WriterFn(BufferQueue& queue, int fd)
{
    while (true) {
        queue.Pop(writeBuffer);
        if (writeBuffer.size() == 0) {
            break;
        }
        ssize_t result = write(fd, writeBuffer.data(), writeBuffer.size());
        if (result != writeBuffer.size()) {
            LOG_ERROR("Error writing to file");
            close(fd);
            throw runtime_error(StringFormat("error happen when writing file. "));
        }
        writeBuffer.clear();
    }
}

void Checkpoint::WriteDataset(CkptTransData& transData,
                              int fd,
                              size_t writeSize,
                              CkptDataType dataType,
                              size_t idx)
{
    ssize_t result = 0;
    if (int32TransSet.find(dataType) != int32TransSet.end()) {
        result = write(fd, reinterpret_cast<const char*>(transData.int32Arr.data()) + idx, writeSize);
    } else if (int64TransSet.find(dataType) != int64TransSet.end()) {
        result = write(fd, reinterpret_cast<const char*>(transData.int64Arr.data()) + idx, writeSize);
    } else if (dataType == CkptDataType::ATTRIBUTE) {
        result = write(fd, reinterpret_cast<const char*>(transData.attribute.data()) + idx, writeSize);
    }

    if (result != writeSize) {
        close(fd);
        LOG_ERROR("Error writing to file, please check the disk buffer or temporary folder space or file permissions!");
        throw runtime_error(StringFormat("error happen when write file. "));
    }
}

void Checkpoint::FillToBuffer(BufferQueue& queue, const char* data, size_t dataSize)
{
    size_t dataIdx = 0;
    while (dataIdx < dataSize) {
        size_t remainingSpace = BUFFER_SIZE - buffer.size();
        if (dataSize - dataIdx <= remainingSpace) {
            buffer.insert(buffer.cend(), data + dataIdx, data + dataSize);
            return;
        } else {
            buffer.insert(buffer.cend(), data + dataIdx, data + dataIdx + remainingSpace);
            queue.Push(std::move(buffer));
            if (buffer.capacity() < BUFFER_SIZE) {
                buffer.reserve(BUFFER_SIZE);
            }
            dataIdx += remainingSpace;
        }
    }
}

void Checkpoint::LoadProcess(CkptData& ckptData)
{
    for (const auto& dataHandler : dataHandlers) {
        vector<string> embNames {};
        vector<string> dirNames { dataHandler->GetDirNames() };
        vector<CkptDataType> saveDataTypes { dataHandler->GetDataTypes() };
        GetUpperLayerLoadDir(dirNames);
        if (find(dirNames.begin(), dirNames.end(), ssdSymbol) != dirNames.end()) {
            embNames = GetTableLayerLoadDir();
        } else {
            embNames = GetEmbedTableNames();
        }
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
        if (embInfo.isSave) {
            loadTableNames.push_back(embInfo.name);
        }
    }

    return loadTableNames;
}

vector<string> Checkpoint::GetTableLayerLoadDir()
{
    vector<string> loadTableDir;
    auto dir { opendir(innerDirPath.c_str()) };
    struct dirent* en;
    if (dir != nullptr) {
        int fileNum = 0;
        while ((en = readdir(dir)) != nullptr) {
            if (fileNum > MAX_FILE_NUM) {
                closedir(dir);
                throw std::runtime_error("The number of files has exceeded the limit " + std::to_string(MAX_FILE_NUM));
            }
            if (strncmp(en->d_name, currDir.c_str(), strlen(currDir.c_str())) != 0 &&
                strncmp(en->d_name, prevDir.c_str(), strlen(prevDir.c_str())) != 0) {
                loadTableDir.emplace_back(en->d_name);
            }
            fileNum++;
        }
        closedir(dir);
    } else {
        LOG_WARN("when loading data in ssd, there are no table files.");
    }
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

            auto datasetDir { datasetPath + dirSeparator + datasetName + to_string(rankId) + dataFileType };
            auto attributeDir { datasetPath + dirSeparator + datasetName + to_string(rankId) + attribFileType };

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

            // load embedding when use dynamic expansion is open
            if ((saveDataType == CkptDataType::NDDR_FEATMAP) && useDynamicExpansion)  {
                auto embedPath { dataDir + dirSeparator + "key_embedding" };
                auto embedDatasetDir { embedPath + dirSeparator + datasetName + to_string(rankId) + dataFileType };
                LOG_DEBUG("====Start loading embedding data from: {}", datasetDir);
                ReadEmbedding(transData, embedDatasetDir, embName);
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

    std::ifstream readFile;
    readFile.open(dataDir.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    size_t datasetSize = static_cast<size_t>(readFile.tellg());
    readFile.seekg(0, std::ios::beg);
    try {
        ValidateReadFile(dataDir, datasetSize);
    } catch (const std::invalid_argument& e) {
        readFile.close();
        throw runtime_error(Logger::Format("Invalid read file path: {}", e.what()));
    }

    if (datasetSize % dataElmtBytes > 0) {
        LOG_DEBUG("data is missing or incomplete in load file: {}", dataDir);
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
            try {
                ReadDataset(transData, readFile, readSize, dataType, idx);
            } catch (std::exception& e) {
                readFile.close();
                throw runtime_error(StringFormat("error happen when reading data from file: %s", e.what()));
            }
            datasetSize -= readSize;
            idx += readSize;
        }
    } else {
        LOG_DEBUG("unable to open load file: {}", dataDir);
    }

    readFile.close();
}

void Checkpoint::ValidateFile(int fd, const string& dataDir, size_t datasetSize) const
{
    try {
        ValidateReadFile(dataDir, datasetSize);
    } catch (const std::invalid_argument& e) {
        close(fd);
        throw runtime_error(StringFormat("Invalid read file path: %s", e.what()));
    }
}

void Checkpoint::HandleMappedData(char* mappedData, size_t mapRowNum, size_t onceReadByteSize,
                                  vector<vector<float>>& dst, size_t cnt) const
{
#pragma omp parallel for
    for (size_t j = 0; j < mapRowNum; ++j) {
        size_t idx = 0;
        size_t readSize = 0;
        size_t dataCol = onceReadByteSize;
        while (dataCol != 0) {
            if (dataCol > oneTimeReadWriteLen) {
                readSize = oneTimeReadWriteLen;
            } else {
                readSize = dataCol;
            }

            errno_t err = memcpy_s(dst[cnt + j].data() + idx, readSize,
                                   mappedData + j * onceReadByteSize + idx, readSize);
            if (err != 0) {
                throw std::runtime_error("Error execution memcpy_s: " + std::to_string(err));
            }
            dataCol -= readSize;
            idx += readSize;
        }
    }
}

void Checkpoint::CalculateMapSize(off_t fileSize, size_t& mapByteSize, size_t& mapRowNum, size_t onceReadByteSize) const
{
    // 每次映射的字节数
    mapByteSize = MAP_BYTE_SIZE;
    // 确保mapByteSize是onceReadByteSize和pageSize的整数倍，确保每次映射的offset是页大小的整数倍
    size_t pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize == -1) {
        throw std::runtime_error("Failed to get page size: " + std::string(strerror(errno)));
    }
    size_t lcmVal = std::lcm(onceReadByteSize, pageSize);
    mapByteSize = (mapByteSize / lcmVal) * lcmVal;

    // 如果文件大小小于每次映射的字节数，则一次性映射，映射大小不是页大小整数倍的时候，mmap会自动向上取整，额外的字节会初始化成零
    if (fileSize <= mapByteSize) {
        mapByteSize = fileSize;
    }

    mapRowNum = mapByteSize / onceReadByteSize;
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
    auto embDataOuterSize = transData.attribute.at(attribEmbDataOuterIdx);
    if (embDataOuterSize <= 0 || embDataOuterSize > MAX_VOCABULARY_SIZE) {
        throw runtime_error(StringFormat("Invalid embDataOuterSize :%d", embDataOuterSize).c_str());
    }

    int fd = open(dataDir.c_str(), O_RDONLY);
    if (fd == -1) {
        throw runtime_error(StringFormat("Failed to open file: %s", dataDir).c_str());
    }

    off_t fileSize = lseek(fd, 0, SEEK_END);
    if (fileSize == 0) {
        close(fd);
        throw runtime_error(StringFormat("emb data file's size is 0").c_str());
    }

    size_t datasetSize = fileSize;
    ValidateFile(fd, dataDir, datasetSize);

    if (datasetSize % embDataOuterSize > 0 || datasetSize % dataElmtBytes > 0) {
        LOG_ERROR("data is missing or incomplete in load file: {}", dataDir);
        close(fd);
        throw runtime_error("unable to load EMB_DATA cause wrong-format saved emb data");
    }

    auto loadHostEmbs = ckptData.hostEmbs;
    auto& dst = (*loadHostEmbs)[embName].embData;
    dst.reserve(embDataOuterSize);

    auto onceReadByteSize { datasetSize / embDataOuterSize };

    size_t mapByteSize;
    size_t mapRowNum;
    CalculateMapSize(fileSize, mapByteSize, mapRowNum, onceReadByteSize);

    off_t offset = 0;
    size_t remainBytes = fileSize;

    for (size_t i = 0; i < embDataOuterSize; i += mapRowNum) {
        // 如果剩余字节数小于每次映射的字节数，则更新每次映射的字节数和行数
        if (remainBytes < mapByteSize) {
            mapByteSize = remainBytes;
            mapRowNum = mapByteSize / onceReadByteSize;
        }

        void* tempMappedData = mmap(NULL, mapByteSize, PROT_READ, MAP_PRIVATE, fd, offset);
        if (tempMappedData == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Failed to map file: " + dataDir + ", errno: " + std::to_string(errno));
        }
        char* mappedData = static_cast<char*>(tempMappedData);

        // 处理映射的数据
        try {
            HandleMappedData(mappedData, mapRowNum, onceReadByteSize, dst, i);
        } catch (const std::runtime_error& e) {
            close(fd);
            munmap(mappedData, mapByteSize);
            throw runtime_error(StringFormat("handle mapped data error: %s", e.what()));
        }
        munmap(mappedData, mapByteSize);

        offset += mapByteSize;
        remainBytes -= mapByteSize;
    }

    close(fd);
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
        readFile.read(reinterpret_cast<char*>(transData.int32Arr.data()) + idx, readSize);
    } else if (int64TransSet.find(dataType) != int64TransSet.end()) {
        readFile.read(reinterpret_cast<char*>(transData.int64Arr.data()) + idx, readSize);
    } else if (floatTransSet.find(dataType) != floatTransSet.end()) {
        readFile.read(reinterpret_cast<char*>(transData.floatArr.data()) + idx, readSize);
    } else if (dataType == CkptDataType::ATTRIBUTE) {
        readFile.read(reinterpret_cast<char*>(transData.attribute.data()) + idx, readSize);
    }
}
