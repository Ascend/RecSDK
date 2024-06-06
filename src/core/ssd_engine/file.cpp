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

#include "file.h"

#include <stdexcept>

#include "utils/common.h"

using namespace MxRec;

/// 创建新文件实例，包含元数据文件、数据文件
/// \param fileID 文件ID
/// \param fileDir 当前文件目录
File::File(uint64_t fileID, string& fileDir) : fileID(fileID), fileDir(fileDir)
{
    LOG_DEBUG("start init file, fileID:{}", fileID);

    if (!fs::exists(fs::absolute(fileDir))) {
        if (!fs::create_directories(fs::absolute(fileDir))) {
            throw runtime_error("fail to create Save directory");
        }
        try {
            fs::permissions(fileDir, fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec);
        } catch (runtime_error &e) {
            LOG_ERROR("fail to change permission of {}", fileDir.c_str());
            fs::remove_all(fileDir);
            throw;
        }
    }

    // latest file is temporary, unnecessary to check file existence and privilege
    metaFilePath = fs::absolute(fileDir + "/" + to_string(fileID) + ".meta.latest");
    dataFilePath = fs::absolute(fileDir + "/" + to_string(fileID) + ".data.latest");
    localFileMeta.open(metaFilePath, ios::out | ios::trunc | ios::binary);
    if (!localFileMeta.is_open()) {
        throw runtime_error("fail to create meta file");
    }
    try {
        fs::permissions(metaFilePath, fs::perms::owner_read | fs::perms::owner_write | fs::perms::group_read);
    } catch (runtime_error &e) {
        LOG_ERROR("fail to change permission of {}", metaFilePath.c_str());
        fs::remove_all(metaFilePath);
        throw;
    }
    localFileData.open(dataFilePath, ios::out | ios::in | ios::trunc | ios::binary);
    if (!localFileData.is_open()) {
        throw runtime_error("fail to create data file");
    }
    try {
        fs::permissions(dataFilePath, fs::perms::owner_read | fs::perms::owner_write | fs::perms::group_read);
    } catch (runtime_error &e) {
        LOG_ERROR("fail to change permission of {}", dataFilePath.c_str());
        fs::remove_all(dataFilePath);
        throw;
    }

    LOG_DEBUG("end init file, fileID:{}", fileID);
}

/// 创建文件实例并加载，从加载路径中读取元数据文件、数据文件，生成临时文件到当前文件目录下
/// \param fileID 文件ID
/// \param loadDir 加载文件的目录
/// \param fileDir 当前文件目录
/// \param step 加载的步数
File::File(uint64_t fileID, string& fileDir, string& loadDir, int step) : fileID(fileID), fileDir(fileDir)
{
    LOG_DEBUG("start init file with load, fileID:{}", fileID);

    fs::path metaFileToLoad = fs::absolute(loadDir + "/" + to_string(fileID) + ".meta." + to_string(step));
    fs::path dataFileToLoad = fs::absolute(loadDir + "/" + to_string(fileID) + ".data." + to_string(step));
    if (!fs::exists(metaFileToLoad)) {
        throw invalid_argument("meta file not found while loading");
    }
    if (!fs::exists(dataFileToLoad)) {
        throw invalid_argument("data file not found while loading");
    }

    ValidateReadFile(metaFileToLoad, fs::file_size(metaFileToLoad));
    ValidateReadFile(dataFileToLoad, fs::file_size(dataFileToLoad));

    // latest file is temporary, unnecessary to check file existence and privilege
    metaFilePath = fs::absolute(fileDir + "/" + to_string(fileID) + ".meta.latest");
    dataFilePath = fs::absolute(fileDir + "/" + to_string(fileID) + ".data.latest");
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
    try {
        fs::permissions(metaFilePath, fs::perms::owner_read | fs::perms::owner_write | fs::perms::group_read);
    } catch (runtime_error &e) {
        LOG_ERROR("fail to change permission of {}", metaFilePath.c_str());
        fs::remove_all(metaFilePath);
        throw;
    }
    localFileData.open(dataFilePath, ios::out | ios::in | ios::binary);
    if (!localFileData.is_open()) {
        throw runtime_error("fail to Load latest data file");
    }
    try {
        fs::permissions(dataFilePath, fs::perms::owner_read | fs::perms::owner_write | fs::perms::group_read);
    } catch (runtime_error &e) {
        LOG_ERROR("fail to change permission of {}", dataFilePath.c_str());
        fs::remove_all(dataFilePath);
        throw;
    }
    Load();

    LOG_DEBUG("end init file with load, fileID:{}", fileID);
}

File::~File()
{
    localFileMeta.close();
    localFileData.close();

    // should call Save before exit program, temporary data will be removed.
    fs::remove(metaFilePath);
    fs::remove(dataFilePath);
}

bool File::IsKeyExist(emb_cache_key_t key) const
{
    auto it = keyToOffset.find(key);
    return !(it == keyToOffset.end());
}

void File::InsertEmbeddings(vector<emb_cache_key_t>& keys, vector<vector<float>>& embeddings)
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
        if (embSize > maxEmbSize) {
            throw invalid_argument("embedding size too large");
        }
        localFileData.write(reinterpret_cast<char const *>(&embSize), sizeof(embSize));
        localFileData.write(reinterpret_cast<char const *>(embeddings[i].data()), embSize * sizeof(float));

        auto pos = localFileData.tellp();
        if (pos == -1) {
            throw runtime_error("can't get file position pointer, write data failed");
        }
        lastWriteOffset = offset_t(pos);
    }
    dataCnt += dLen;
}

vector<vector<float>> File::FetchEmbeddings(vector<emb_cache_key_t>& keys)
{
    vector<vector<float>> ret;
    for (emb_cache_key_t k: keys) {
        auto it = keyToOffset.find(k);
        if (it == keyToOffset.end()) {
            throw invalid_argument("key not exist");
        }
        localFileData.seekg(it->second); // for fstream, this moves the file position pointer (both put and get)
        if (localFileData.fail()) {
            throw runtime_error("can't move file position pointer, read data failed");
        }

        uint64_t embSize;
        localFileData.read(reinterpret_cast<char *>(&embSize), sizeof(embSize));
        if (localFileData.fail()) {
            throw invalid_argument("read embedding size failed, file may broken");
        }
        if (embSize > maxEmbSize) {
            throw invalid_argument("embedding size too large, file may broken");
        }

        vector<float> tmp;
        tmp.resize(embSize);
        localFileData.read(reinterpret_cast<char *>(tmp.data()), tmp.size() * sizeof(float));
        ret.emplace_back(tmp);
    }
    return ret;
}

void File::DeleteEmbedding(emb_cache_key_t key)
{
    if (!IsKeyExist(key)) {
        return;
    }
    keyToOffset.erase(key);
    staleDataCnt += 1;
}

void File::Save(const string& saveDir, int step)
{
    LOG_DEBUG("start save file at step:{}, fileID:{}", step, fileID);

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

    LOG_DEBUG("save latest meta file at step:{}, fileID:{}", step, fileID);
    if (!fs::copy_file(metaFilePath, metaFileToSave)) {
        throw runtime_error("fail to Save latest meta");
    }

    // re-open new meta file for next saving
    localFileMeta.open(metaFilePath, ios::out | ios::trunc | ios::binary);
    if (!localFileMeta.is_open()) {
        throw runtime_error("fail to re-open meta file");
    }

    // Save data
    LOG_DEBUG("save latest data file at step:{}", step);
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

    LOG_DEBUG("end save file at step:{}, fileID:{}", step, fileID);
}

void File::Load()
{
    // file already validate and open in instantiation
    LOG_DEBUG("start reading meta file, fileID:{}", fileID);
    emb_cache_key_t key;
    offset_t offset;
    do {
        localFileMeta.read(reinterpret_cast<char*>(&key), KEY_DATA_LEN);
        if (!localFileMeta.eof() && localFileMeta.fail()) {
            throw invalid_argument("file broken while reading key");
        }

        localFileMeta.read(reinterpret_cast<char*>(&offset), OFFSET_DATA_LEN);
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

    LOG_DEBUG("end reading meta file, fileID:{}", fileID);
}

vector<emb_cache_key_t> File::GetKeys()
{
    vector<emb_cache_key_t> ret;
    for (auto item: keyToOffset) {
        ret.push_back(item.first);
    }
    return ret;
}

uint64_t File::GetDataCnt() const
{
    return dataCnt;
}

uint64_t File::GetFileID() const
{
    return fileID;
}

uint64_t File::GetStaleDataCnt() const
{
    return staleDataCnt;
}

void File::InsertEmbeddingsByAddr(vector<emb_cache_key_t>& keys, vector<float*>& embeddingsAddr,
                                  uint64_t extEmbeddingSize)
{
    if (keys.size() != embeddingsAddr.size()) {
        throw invalid_argument("keys' length not equal to embeddings' length");
    }

    size_t dLen = keys.size();
    for (size_t i = 0; i < dLen; ++i) {
        if (embeddingsAddr[i] == nullptr) {
            throw invalid_argument("Null pointer found in embeddingsAddr");
        }
    }

    localFileData.seekp(lastWriteOffset);  // always set pointer to buffer end in case reading happened before

    for (size_t i = 0; i < dLen; ++i) {
        if (IsKeyExist(keys[i])) {
            staleDataCnt++;
        }
        keyToOffset[keys[i]] = lastWriteOffset;

        if (extEmbeddingSize > maxEmbSize) {
            throw invalid_argument("embedding size too large");
        }
        localFileData.write(reinterpret_cast<char const*>(&extEmbeddingSize), sizeof(extEmbeddingSize));
        localFileData.write(reinterpret_cast<char const*>(embeddingsAddr[i]), extEmbeddingSize * sizeof(float));

        auto pos = localFileData.tellp();
        if (pos == -1) {
            throw runtime_error("can't get file position pointer, write data failed");
        }
        lastWriteOffset = offset_t(pos);
    }
    dataCnt += dLen;
}
