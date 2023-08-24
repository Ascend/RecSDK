/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */
#include "file.h"

#include <stdexcept>

#include "utils/common.h"

using namespace MxRec;

/// 创建新文件实例，包含元数据文件、数据文件
/// \param fileID 文件ID
/// \param saveDir 保存文件夹的路径
File::File(uint64_t fileID, string &saveDir) : fileID(fileID), saveDir(saveDir)
{
    VLOG(GLOG_DEBUG) << StringFormat("start init file, fileID:%llu", fileID);

    if (!fs::exists(fs::absolute(saveDir))) {
        if (!fs::create_directories(fs::absolute(saveDir))) {
            throw runtime_error("fail to create Save directory");
        }
    }

    metaFilePath = fs::absolute(saveDir + "/" + to_string(fileID) + ".meta.latest");
    dataFilePath = fs::absolute(saveDir + "/" + to_string(fileID) + ".data.latest");
    localFileMeta.open(metaFilePath, ios::out | ios::trunc | ios::binary);
    if (!localFileMeta.is_open()) {
        throw runtime_error("fail to create meta file");
    }
    fs::permissions(metaFilePath, fs::perms::owner_read | fs::perms::owner_write);
    localFileData.open(dataFilePath, ios::out | ios::in | ios::trunc | ios::binary);
    if (!localFileData.is_open()) {
        throw runtime_error("fail to create data file");
    }
    fs::permissions(dataFilePath, fs::perms::owner_read | fs::perms::owner_write);

    VLOG(GLOG_DEBUG) << StringFormat("end init file, fileID:%llu", fileID);
}

/// 创建文件实例并加载，从保存路径中读取元数据文件、数据文件
/// \param fileID 文件ID
/// \param saveDir 保存文件夹的路径
/// \param step 加载的步数
File::File(uint64_t fileID, string &saveDir, int step) : fileID(fileID), saveDir(saveDir)
{
    VLOG(GLOG_DEBUG) << StringFormat("start init file with load, fileID:%llu", fileID);

    fs::path metaFileToLoad = fs::absolute(saveDir + "/" + to_string(fileID) + ".meta." + to_string(step));
    fs::path dataFileToLoad = fs::absolute(saveDir + "/" + to_string(fileID) + ".data." + to_string(step));
    if (!fs::exists(metaFileToLoad)) {
        throw invalid_argument("meta file not found while loading");
    }
    if (!fs::exists(dataFileToLoad)) {
        throw invalid_argument("data file not found while loading");
    }

    ValidateReadFile(metaFileToLoad, fs::file_size(metaFileToLoad));
    ValidateReadFile(dataFileToLoad, fs::file_size(dataFileToLoad));

    metaFilePath = fs::absolute(saveDir + "/" + to_string(fileID) + ".meta.latest");
    dataFilePath = fs::absolute(saveDir + "/" + to_string(fileID) + ".data.latest");
    fs::remove(metaFilePath);
    fs::remove(dataFilePath);

    if (!fs::copy_file(metaFileToLoad, metaFilePath)) {
        throw runtime_error("fail to create latest meta file");
    }
    if (!fs::copy_file(dataFileToLoad, dataFilePath)) {
        throw runtime_error("fail to create latest data file");
    }

    localFileMeta.open(metaFilePath, ios::in | ios::binary);
    if (!localFileMeta.is_open()) {
        throw runtime_error("fail to Load latest meta file");
    }
    fs::permissions(metaFilePath, fs::perms::owner_read | fs::perms::owner_write);
    localFileData.open(dataFilePath, ios::out | ios::in | ios::binary);
    if (!localFileData.is_open()) {
        throw runtime_error("fail to Load latest data file");
    }
    fs::permissions(dataFilePath, fs::perms::owner_read | fs::perms::owner_write);
    Load();

    VLOG(GLOG_DEBUG) << StringFormat("end init file with load, fileID:%llu", fileID);
}

File::~File()
{
    localFileMeta.close();
    localFileData.close();

    // should call Save before exit program, temporary data will be removed.
    fs::remove(metaFilePath);
    fs::remove(dataFilePath);
}

bool File::IsKeyExist(emb_key_t key)
{
    auto it = keyToOffset.find(key);
    return !(it == keyToOffset.end());
}

void File::InsertEmbeddings(vector<emb_key_t> &keys, vector<vector<float>> &embeddings)
{
    if (keys.size() != embeddings.size()) {
        throw invalid_argument("keys' length not equal to embeddings' length");
    }

    localFileData.seekp(lastWriteOffset); // always set pointer to buffer end in case reading happened before

    size_t dLen = keys.size();
    for (size_t i = 0; i < dLen; ++i) {
        if (IsKeyExist(keys[i])) {
            staleDataCnt++;
        }
        keyToOffset[keys[i]] = lastWriteOffset;

        uint64_t embSize = embeddings[i].size();
        localFileData.write(reinterpret_cast<char const *>(&embSize), sizeof(embSize));
        localFileData.write(reinterpret_cast<char const *>(embeddings[i].data()),
                            embeddings[i].size() * sizeof(float));

        auto pos = localFileData.tellp();
        if (pos == -1) {
            throw runtime_error("can't get file position pointer");
        }
        lastWriteOffset = offset_t(pos);
    }
    dataCnt += dLen;
}

vector<vector<float>> File::FetchEmbeddings(vector<emb_key_t> &keys)
{
    vector<vector<float>> ret;
    for (emb_key_t k: keys) {
        auto it = keyToOffset.find(k);
        if (it == keyToOffset.end()) {
            throw invalid_argument("key not exist");
        }
        localFileData.seekg(it->second); // for fstream, this moves the file position pointer (both put and get)
        if (localFileData.fail()) {
            throw runtime_error("can't move file position pointer");
        }

        uint64_t embSize;
        localFileData.read(reinterpret_cast<char *>(&embSize), sizeof(embSize));

        vector<float> tmp;
        tmp.resize(embSize);
        localFileData.read(reinterpret_cast<char *>(tmp.data()), tmp.size() * sizeof(float));
        ret.emplace_back(tmp);
    }
    return ret;
}

void File::DeleteEmbedding(emb_key_t key)
{
    if (!IsKeyExist(key)) {
        return;
    }
    keyToOffset.erase(key);
    staleDataCnt += 1;
}

void File::Save(int step)
{
    VLOG(GLOG_DEBUG) << StringFormat("start save file at step:%d, fileID:%llu", step, fileID);

    // write current meta into meta file
    for (auto [key, offset]: keyToOffset) {
        localFileMeta.write(reinterpret_cast<char const *>(&key), sizeof(key));
        localFileMeta.write(reinterpret_cast<char const *>(&offset), sizeof(offset));
    }
    // flush not guarantee data already written into disk, must call close to force flush and wait
    localFileMeta.flush();
    if (localFileMeta.fail()) {
        throw runtime_error("fail to save latest meta");
    }
    localFileMeta.close();

    fs::path metaFileToSave = fs::absolute(saveDir + "/" + to_string(fileID) + ".meta." + to_string(step));
    if (fs::exists(metaFileToSave)) {
        throw invalid_argument("fail to save latest meta, file already exist");
    }

    VLOG(GLOG_DEBUG) << StringFormat("save latest meta file at step:%d, fileID:%llu", step, fileID);
    if (!fs::copy_file(metaFilePath, metaFileToSave)) {
        throw runtime_error("fail to Save latest meta");
    }

    // re-open new meta file for next saving
    localFileMeta.open(metaFilePath, ios::out | ios::trunc | ios::binary);
    if (!localFileMeta.is_open()) {
        throw runtime_error("fail to re-open meta file");
    }

    // Save data
    VLOG(GLOG_DEBUG) << StringFormat("save latest data file at step:%d", step);
    localFileData.flush();
    if (localFileData.fail()) {
        throw runtime_error("fail to Save data");
    }
    localFileData.close();

    fs::path dataFileToSave = fs::absolute(saveDir + "/" + to_string(fileID) + ".data." + to_string(step));
    if (fs::exists(dataFileToSave)) {
        throw invalid_argument("fail to save latest data, file already exist");
    }
    if (!fs::copy_file(dataFilePath, dataFileToSave)) {
        throw runtime_error("fail to Save latest data");
    }

    // re-open data file for other operation
    localFileData.open(dataFilePath, ios::out | ios::in | ios::app | ios::binary);
    if (!localFileData.is_open()) {
        throw runtime_error("fail to re-open data file");
    }

    VLOG(GLOG_DEBUG) << StringFormat("end save file at step:%d, fileID:%llu", step, fileID);
}

void File::Load()
{
    // file already validate and open in instantiation
    VLOG(GLOG_DEBUG) << StringFormat("start reading meta file, fileID:%llu", fileID);
    emb_key_t key;
    offset_t offset;
    do {
        localFileMeta.read(reinterpret_cast<char *>(&key), keyDataLen);
        if (!localFileMeta.eof() && localFileMeta.fail()) {
            throw invalid_argument("file broken while reading key");
        }

        localFileMeta.read(reinterpret_cast<char *>(&offset), offsetDataLen);
        if (!localFileMeta.eof() && localFileMeta.fail()) {
            throw invalid_argument("file broken while reading offset");
        }
        keyToOffset[key] = offset;
        dataCnt += 1;

        // try reading one byte to see if pointer reach end
        localFileMeta.get();
        if (localFileMeta.eof()) {
            break;
        }
        localFileMeta.unget();
    } while (!localFileMeta.eof());
    localFileMeta.close();

    // re-open new meta file for next saving
    localFileMeta.open(metaFilePath, ios::out | ios::trunc | ios::binary);
    if (!localFileMeta.is_open()) {
        throw runtime_error("fail to re-open meta file");
    }

    VLOG(GLOG_DEBUG) << StringFormat("end reading meta file, fileID:%llu", fileID);
}

vector<emb_key_t> File::GetKeys()
{
    vector<emb_key_t> ret;
    for (auto item: keyToOffset) {
        ret.push_back(item.first);
    }
    return ret;
}

uint64_t File::GetDataCnt()
{
    return dataCnt;
}

uint64_t File::GetFileID()
{
    return fileID;
}

uint64_t File::GetStaleDataCnt()
{
    return staleDataCnt;
}
