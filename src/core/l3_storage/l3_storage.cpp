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

#include "l3_storage.h"

using MxRec::L3Storage;

L3Storage::L3Storage() {}

L3Storage::~L3Storage() {}

bool L3Storage::IsTableExist(const string& tableName)
{
    return false;
}

bool L3Storage::IsKeyExist(const string& tableName, emb_cache_key_t key)
{
    return false;
}

void L3Storage::CreateTable(const string& tableName, vector<string> savePaths, uint64_t maxTableSize) {}

int64_t L3Storage::GetTableAvailableSpace(const string& tableName)
{
    return 0;
}

void L3Storage::InsertEmbeddingsByAddr(const string& tableName, vector<emb_cache_key_t>& keys,
                                            vector<float*>& embeddingsAddr, uint64_t extEmbeddingSize)
{
}

void L3Storage::DeleteEmbeddings(const string& tableName, vector<emb_cache_key_t>& keys) {}

vector<vector<float>> L3Storage::FetchEmbeddings(const string& tableName, vector<emb_cache_key_t>& keys)
{
    return vector<vector<float>>();
}

void L3Storage::Save(int step) {}

void L3Storage::Load(const string& tableName, vector<string> savePaths, uint64_t maxTableSize, int step) {}

void L3Storage::Start() {}

void L3Storage::Stop() {}

int64_t L3Storage::GetTableUsage(const string& tableName)
{
    return 0;
}

vector<std::pair<string, vector<emb_cache_key_t>>> L3Storage::ExportTableKey()
{
    return vector<std::pair<string, vector<emb_cache_key_t>>>();
}
