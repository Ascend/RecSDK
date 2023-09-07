/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */


#include "table.h"
#include "utils/common.h"

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
    curTablePath = fs::absolute(savePaths.at(curSavePathIdx) + "/" + saveDirPrefix + g_rankId + "/" + name).string();
    if (!fs::exists(curTablePath) && !fs::create_directories(curTablePath)) {
        throw runtime_error("fail to create table directory");
    }
    LOG(INFO) << StringFormat("create table:%s at path:%s", name.c_str(), curTablePath.c_str());
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
    bool isMetaFileFound = false;
    for (const string &dirPath: saveDirs) {
        auto metaFilePath = fs::absolute(
            dirPath + "/" + saveDirPrefix + g_rankId + "/" + name + "/" + name + ".meta." + to_string(step)).string();
        if (!fs::exists(metaFilePath)) {
            continue;
        }
        Load(metaFilePath, step);
        isMetaFileFound = true;
        break;
    }
    if (!isMetaFileFound) {
        throw invalid_argument("table meta file not found");
    }

    // always use first path to save until it's full
    curTablePath = fs::absolute(savePaths.at(curSavePathIdx) + "/" + saveDirPrefix + g_rankId + "/" + name).string();
    LOG(INFO) << StringFormat("load table:%s done. try store at path:%s", name.c_str(), curTablePath.c_str());
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
    LOG(INFO) << StringFormat("start save table:%s, at step:%d", name.c_str(), step);
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
        f->Save(step);
    }

    metaFile.flush();
    if (metaFile.fail()) {
        throw runtime_error("fail to Save table meta file");
    }

    metaFile.close();
    LOG(INFO) << StringFormat("end save table:%s, at step:%d", name.c_str(), step);
}

/// 根据元数据加载data文件
/// \param metaFile 元数据文件
/// \param step 加载的步数
void Table::LoadDataFileSet(const shared_ptr<fstream> &metaFile, int step)
{
    LOG(INFO) << StringFormat("table:%s, start load data file", name.c_str());
    uint64_t fileCnt;
    metaFile->read(reinterpret_cast<char *>(&fileCnt), sizeof(fileCnt));
    uint64_t fileID;
    uint64_t fidSize = sizeof(fileID);
    for (uint64_t i = 0; i < fileCnt; ++i) {
        metaFile->read(reinterpret_cast<char *>(&fileID), fidSize);
        if (fileID > curMaxFileID) {
            curMaxFileID = fileID;
        }

        bool isFileFound = false;
        shared_ptr<File> tmp;
        for (const string &p: savePaths) {
            // try to find data file from each path
            string dataPath = p + "/" + saveDirPrefix + g_rankId + "/" + name;
            try {
                tmp = make_shared<File>(fileID, dataPath, step);
                fileSet.insert(tmp);
                isFileFound = true;
                break;
            } catch (invalid_argument &e) {
                // do nothing because file may in other path
            }
        }
        if (!isFileFound) {
            throw invalid_argument("data file not found");
        }

        auto keys = tmp->GetKeys();
        totalKeyCnt += keys.size();
        if (totalKeyCnt > maxTableSize) {
            throw invalid_argument("table size too small, key quantity exceed while loading data");
        }

        for (emb_key_t k: keys) {
            if (keyToFile.find(k) != keyToFile.end()) {
                throw invalid_argument(
                    "find duplicate key in files, compaction already done before saving, file may broken or modified");
            }
            keyToFile[k] = tmp;
        }
    }
    curMaxFileID += 1;
}


void Table::Load(const string &metaFilePath, int step)
{
    ValidateReadFile(metaFilePath, fs::file_size(metaFilePath));

    shared_ptr<fstream> metaFile = make_shared<fstream>();
    metaFile->open(metaFilePath, ios::in | ios::binary);
    LOG(INFO) << StringFormat("table:%s, load meta file from path:%s", name.c_str(), metaFilePath.c_str());
    if (!metaFile->is_open()) {
        throw invalid_argument("fail to open meta");
    }

    // Load table name and validate
    uint32_t nameSize;
    metaFile->read(reinterpret_cast<char *>(&nameSize), sizeof(nameSize));
    if (nameSize > maxNameSize) {
        throw invalid_argument("table name too large, file may broken");
    }
    char *tmpArr = new char[nameSize + 1];
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
    LOG(INFO) << StringFormat("table:%s, end load data file", name.c_str());
}

void Table::InsertEmbeddingsInner(vector<emb_key_t> &keys, vector<vector<float>> &embeddings)
{
    if (totalKeyCnt > maxTableSize) {
        throw invalid_argument("table size too small, key quantity exceed while loading data");
    }

    if (curFile == nullptr || (curFile != nullptr && curFile->GetDataCnt() >= maxDataNumInFile)) {
        // leave diskFreeSpaceThreshold % space for each disk
        while (true) {
            fs::space_info si = fs::space((curTablePath));
            if ((double(si.free) / double(si.capacity)) > diskFreeSpaceThreshold) {
                break;
            }

            curSavePathIdx += 1;
            if (curSavePathIdx >= savePaths.size()) {
                throw runtime_error("all disk's space not enough");
            }
            curTablePath = savePaths[curSavePathIdx];
            LOG(INFO) << StringFormat(
                "current data path's free space less than %f, try next path:%s",
                diskFreeSpaceThreshold, curTablePath.c_str()
            );
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
        auto it = keyToFile.find(keys[i]);
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
        shared_ptr<File> f;
        vector<long> batchKeys;
        vector<size_t> batchIdx;
        tie(f, batchKeys, batchIdx) = item;
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

    VLOG(GLOG_DEBUG) << StringFormat("table:%s, start compact", name.c_str());

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
    VLOG(GLOG_DEBUG) << StringFormat("table:%s, end compact", name.c_str());
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
