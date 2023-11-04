/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */


#include "table.h"

using namespace MxRec;

/// 创建新表
/// \param name 表名
/// \param savePaths 表的存储路径
/// \param maxTableSize 表的最大空间，按key数量计
/// \param compactThreshold 表的压缩阈值，当无效数据占比超阈值时，文件会被清理
Table::Table(const string &name, vector<string> &savePaths, uint64_t maxTableSize, double compactThreshold)
    : name(name),
      savePaths(savePaths),
      maxTableSize(maxTableSize),
      compactThreshold(compactThreshold)
{
    curTablePath = fs::absolute(savePaths.at(curSavePathIdx) + "/" +
        saveDirPrefix + GlogConfig::gRankId + "/" + name).string();
    if (!fs::exists(curTablePath) && !fs::create_directories(curTablePath)) {
        throw runtime_error("fail to create table directory");
    }
    LOG_INFO("create table:{} at path:{}", name, curTablePath);
}

/// 加载表
/// \param name 表名
/// \param savePaths 表的存储路径
/// \param maxTableSize 表的最大空间，按key数量计
/// \param compactThreshold 表的压缩阈值，当无效数据占比超阈值时，文件会被清理
/// \param step 加载的步数
Table::Table(const string &name, vector<string> &saveDirs, uint64_t maxTableSize, double compactThreshold, int step)
    : name(name),
      savePaths(saveDirs),
      maxTableSize(maxTableSize),
      compactThreshold(compactThreshold)
{
    // always use first path to save until it's full
    curTablePath = fs::absolute(savePaths.at(curSavePathIdx) + "/" +
        saveDirPrefix + GlogConfig::gRankId + "/" + name).string();
    if (!fs::exists(curTablePath) && !fs::create_directories(curTablePath)) {
        throw runtime_error("fail to create table directory");
    }

    bool isMetaFileFound = false;
    for (const string &dirPath: saveDirs) {
        auto metaFilePath = fs::absolute(
            dirPath + "/" + saveDirPrefix + GlogConfig::gRankId + "/" +
            name + "/" + name + ".meta." + to_string(step)).string();
        if (!fs::exists(metaFilePath)) {
            continue;
        }
        Load(metaFilePath, step);
        isMetaFileFound = true;
        break;
    }
    if (!isMetaFileFound) {
        throw invalid_argument(StringFormat("table:%s meta file not found", name.c_str()));
    }

    LOG_INFO("load table:{} done. try store at path:{}", name, curTablePath);
}

bool Table::IsKeyExist(emb_key_t key)
{
    lock_guard<mutex> guard(rwLock);
    auto it = keyToFile.find(key);
    return !(it == keyToFile.end());
}

void Table::InsertEmbeddings(vector<emb_key_t> &keys, vector<vector<float>> &embeddings)
{
    lock_guard<mutex> guard(rwLock);
    InsertEmbeddingsInner(keys, embeddings);
}

vector<vector<float>> Table::FetchEmbeddings(vector<emb_key_t> &keys)
{
    lock_guard<mutex> guard(rwLock);
    return FetchEmbeddingsInner(keys);
}


void Table::DeleteEmbeddings(vector<emb_key_t> &keys)
{
    lock_guard<mutex> guard(rwLock);
    DeleteEmbeddingsInner(keys);
}

void Table::Save(int step)
{
    LOG_INFO("start save table:{}, at step:{}", name, step);
    Compact(true);

    lock_guard<mutex> guard(rwLock);
    auto metaFilePath = fs::absolute(curTablePath + "/" + name + ".meta" + "." + to_string(step));
    if (fs::exists(metaFilePath)) {
        throw invalid_argument("fail to save table meta, file already exist");
    }

    fstream metaFile;
    metaFile.open(metaFilePath, ios::out | ios::trunc | ios::binary);
    if (!metaFile.is_open()) {
        throw runtime_error("fail to create table meta file");
    }
    fs::permissions(metaFilePath, fs::perms::owner_read | fs::perms::owner_write);

    // dump table name
    uint32_t nameSize = static_cast<uint32_t>(name.size());
    metaFile.write(reinterpret_cast<char const *>(&nameSize), sizeof(nameSize));
    metaFile.write(name.c_str(), nameSize);

    // dump file ID
    uint64_t fileCnt = fileSet.size();
    metaFile.write(reinterpret_cast<char const *>(&fileCnt), sizeof(fileCnt));
    for (const auto &f: fileSet) {
        uint64_t fid = f->GetFileID();
        metaFile.write(reinterpret_cast<char const *>(&fid), sizeof(fid));
        SetTablePathToDiskWithSpace();
        if (!fs::exists(curTablePath) && !fs::create_directories(curTablePath)) {
            throw runtime_error("fail to create table directory");
        }
        f->Save(curTablePath, step);
    }

    metaFile.flush();
    if (metaFile.fail()) {
        throw runtime_error("fail to Save table meta file");
    }

    metaFile.close();
    LOG_INFO("end save table:{}, at step:{}", name, step);
}

/// 根据元数据加载data文件
/// \param metaFile 元数据文件
/// \param step 加载的步数
void Table::LoadDataFileSet(const shared_ptr<fstream> &metaFile, int step)
{
    LOG_INFO("table:{}, start load data file", name);
    uint64_t fileCnt;
    metaFile->read(reinterpret_cast<char *>(&fileCnt), sizeof(fileCnt));
    if (metaFile->fail()) {
        throw invalid_argument("fail to read nFile, meta file broken");
    }
    uint64_t fileID;
    uint64_t fidSize = sizeof(fileID);
    for (uint64_t i = 0; i < fileCnt; ++i) {
        metaFile->read(reinterpret_cast<char *>(&fileID), fidSize);
        if (metaFile->fail()) {
            throw invalid_argument("fail to read fileID, meta file broken");
        }
        if (fileID > curMaxFileID) {
            curMaxFileID = fileID;
        }

        shared_ptr<File> loadedFile = nullptr;
        for (const string &p: savePaths) {
            // try to find data file from each path
            string loadPath = p + "/" + saveDirPrefix + GlogConfig::gRankId + "/" + name;
            SetTablePathToDiskWithSpace();
            if (!fs::exists(curTablePath) && !fs::create_directories(curTablePath)) {
                throw runtime_error("fail to create table directory");
            }
            try {
                loadedFile = make_shared<File>(fileID, curTablePath, loadPath, step);
                fileSet.insert(loadedFile);
                break;
            } catch (invalid_argument &e) {
                // do nothing because file may in other path
            }
        }
        if (loadedFile == nullptr) {
            throw invalid_argument("data file not found");
        }

        auto keys = loadedFile->GetKeys();
        totalKeyCnt += keys.size();
        if (totalKeyCnt > maxTableSize) {
            throw invalid_argument("table size too small, key quantity exceed while loading data");
        }

        for (emb_key_t k: keys) {
            if (keyToFile.find(k) != keyToFile.end()) {
                throw invalid_argument(
                    "find duplicate key in files, compaction already done before saving, file may broken or modified");
            }
            keyToFile[k] = loadedFile;
        }
    }
    curMaxFileID += 1;
}


void Table::Load(const string &metaFilePath, int step)
{
    ValidateReadFile(metaFilePath, fs::file_size(metaFilePath));

    shared_ptr<fstream> metaFile = make_shared<fstream>();
    metaFile->open(metaFilePath, ios::in | ios::binary);
    LOG_INFO("table:{}, load meta file from path:{}", name, metaFilePath);
    if (!metaFile->is_open()) {
        throw invalid_argument("fail to open meta");
    }

    // Load table name and validate
    uint32_t nameSize;
    metaFile->read(reinterpret_cast<char *>(&nameSize), sizeof(nameSize));
    if (metaFile->fail()) {
        throw invalid_argument("fail to read table name size");
    }
    if (nameSize > maxNameSize) {
        throw invalid_argument("table name too large, file may broken");
    }
    char tmpArr[nameSize + 1];
    auto ec = memset_s(tmpArr, nameSize + 1, '\0', nameSize + 1);
    if (ec != EOK) {
        throw runtime_error("fail to init table name array");
    }
    metaFile->read(tmpArr, static_cast<long>(nameSize));
    tmpArr[nameSize] = '\0';
    string tbNameInFile = tmpArr;
    if (name != tbNameInFile) {
        throw invalid_argument("table name not match");
    }

    // construct file set
    LoadDataFileSet(metaFile, step);
    metaFile->close();
    if (metaFile->fail()) {
        throw runtime_error("fail to load table");
    }
    LOG_INFO("table:{}, end load data file", name);
}

void Table::InsertEmbeddingsInner(vector<emb_key_t> &keys, vector<vector<float>> &embeddings)
{
    if (totalKeyCnt > maxTableSize) {
        throw invalid_argument("table size too small, key quantity exceed while loading data");
    }

    if (curFile == nullptr || (curFile != nullptr && curFile->GetDataCnt() >= maxDataNumInFile)) {
        SetTablePathToDiskWithSpace();
        if (!fs::exists(curTablePath) && !fs::create_directories(curTablePath)) {
            throw runtime_error("fail to create table directory");
        }
        curFile = make_shared<File>(curMaxFileID, curTablePath);
        fileSet.insert(curFile);
        curMaxFileID++;
    }

    for (emb_key_t k: keys) {
        auto it = keyToFile.find(k);
        if (it != keyToFile.end()) {
            it->second->DeleteEmbedding(k);
            staleDataFileSet.insert(it->second);
            totalKeyCnt -= 1;
        }
        keyToFile[k] = curFile;
    }
    curFile->InsertEmbeddings(keys, embeddings);
    totalKeyCnt += keys.size();
}

vector<vector<float>> Table::FetchEmbeddingsInner(vector<emb_key_t> &keys)
{
    // build mini batch for each file, first element for keys, second for index
    size_t dLen = keys.size();
    unordered_map<shared_ptr<File>, shared_ptr<pair<vector<emb_key_t>, vector<size_t>>>> miniBatch;
    for (size_t i = 0; i < dLen; ++i) {
        auto it = as_const(keyToFile).find(keys[i]);
        if (it == keyToFile.end()) {
            throw invalid_argument(StringFormat("failed to find the key, {key=%d} not exist!", keys[i]));
        }
        if (miniBatch[it->second] == nullptr) {
            miniBatch[it->second] = make_shared<pair<vector<emb_key_t>, vector<size_t>>>();
        }
        miniBatch[it->second]->first.emplace_back(keys[i]);
        miniBatch[it->second]->second.emplace_back(i);
    }

    // must convert map to list to perform parallel query, omp not support to iterate map
    vector<tuple<shared_ptr<File>, vector<emb_key_t>, vector<size_t>>> queryList;
    queryList.reserve(miniBatch.size());
    for (auto [f, info]: miniBatch) {
        queryList.emplace_back(f, info->first, info->second);
    }

    // read in parallel
    vector<vector<float>> ret;
    ret.resize(dLen);
    size_t queryLen = queryList.size();
#pragma omp parallel for num_threads(readThreadNum) default(none) shared(ret, queryLen, queryList)
    for (size_t i = 0; i < queryLen; ++i) {
        tuple item = queryList[i];
        auto [f, batchKeys, batchIdx] = item;
        vector<vector<float>> batchRet = f->FetchEmbeddings(batchKeys);
        size_t batchLen = batchRet.size();
        for (size_t j = 0; j < batchLen; ++j) {
            ret[batchIdx[j]] = batchRet[j];
        }
    }
    return ret;
}

/// 整理数据，将有效数据转移至新文件后，含无效数据的文件将被删除
/// \param fullCompact 是否执行全量数据清理
void Table::Compact(bool fullCompact)
{
    lock_guard<mutex> guard(rwLock);

    if (staleDataFileSet.empty()) {
        return;
    }

    LOG_DEBUG("table:{}, start compact", name);

    vector<shared_ptr<File>> compactFileList;
    for (const auto &f: staleDataFileSet) {
        if (fullCompact) {
            compactFileList.emplace_back(f);
            continue;
        }
        if (double(f->GetDataCnt()) * compactThreshold < double(f->GetStaleDataCnt())) {
            compactFileList.emplace_back(f);
        }
    }

    // always move valid data to new file to avoid repeated compaction
    if (curFile->GetStaleDataCnt() > 0) {
        curFile = make_shared<File>(curMaxFileID, curTablePath);
        fileSet.insert(curFile);
        curMaxFileID++;
    }

    for (const auto &f: compactFileList) {
        staleDataFileSet.erase(f);
        fileSet.erase(f);
        vector<emb_key_t> validKeys = f->GetKeys();
        vector<vector<float>> validEmbs = f->FetchEmbeddings(validKeys);
        InsertEmbeddingsInner(validKeys, validEmbs);
    }
    LOG_DEBUG("table:{}, end compact", name);
}

uint64_t Table::GetTableAvailableSpace()
{
    lock_guard<mutex> guard(rwLock);
    return maxTableSize - totalKeyCnt;
}

void Table::DeleteEmbeddingsInner(vector<emb_key_t> &keys)
{
    for (emb_key_t k: keys) {
        auto it = keyToFile.find(k);
        if (it != keyToFile.end()) {
            it->second->DeleteEmbedding(k);
            staleDataFileSet.insert(it->second);
            keyToFile.erase(k);
            totalKeyCnt -= 1;
        }
    }
}

void Table::SetTablePathToDiskWithSpace()
{
    constexpr int nMaxLoop = 1024;
    int loopCnt = 0;
    while (loopCnt < nMaxLoop) {
        fs::space_info si = fs::space(savePaths.at(curSavePathIdx));
        if ((double(si.available) / double(si.capacity)) > diskAvailSpaceThreshold) {
            break;
        }

        curSavePathIdx += 1;
        if (curSavePathIdx >= savePaths.size()) {
            throw runtime_error("all disk's space not enough");
        }
        curTablePath = fs::absolute(
            savePaths.at(curSavePathIdx) + "/" + saveDirPrefix + GlogConfig::gRankId + "/" + name).string();

        LOG_INFO("current data path's available space less than {}%, try next path:{}",
                 diskAvailSpaceThreshold * convertToPercentage, curTablePath);
        loopCnt += 1;
    }
}

uint64_t Table::GetTableUsage()
{
    lock_guard<mutex> guard(rwLock);
    return totalKeyCnt;
}

