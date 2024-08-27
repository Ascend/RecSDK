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

#ifndef MX_REC_CHECKPOINT_H
#define MX_REC_CHECKPOINT_H

#include <dirent.h>
#include <acl/acl_rt.h>
#include <acl/acl.h>
#include "utils/common.h"
#include "ckpt_data_handler/ckpt_data_handler.h"
#include "file_system/file_system_handler.h"

namespace MxRec {
    using namespace std;

    class Checkpoint {
    public:
        Checkpoint() = default;
        ~Checkpoint() {};

        void SaveModel(string savePath, CkptData& ckptData, RankInfo& mgmtRankInfo, const vector<EmbInfo>& embInfo);
        void LoadModel(string loadPath, CkptData& ckptData, RankInfo& mgmtRankInfo, const vector<EmbInfo>& embInfo,
                       const vector<CkptFeatureType>& featureTypes);

    private:
        std::vector<char> buffer;
        std::vector<char> writeBuffer;
        const string datasetName { "slice_" };
        const string dataFileType { ".data" };
        const string attribFileType { ".attribute" };
        const string dirSeparator { "/" };
        const string ssdSymbol {"SSD"};
        const mode_t dirMode { 0750 };

        const size_t oneTimeReadWriteLen { 32768 }; // 4096 * 8

        const set<CkptDataType> int32TransSet {
            CkptDataType::EMB_INFO,
            CkptDataType::EMB_CURR_STAT,
            CkptDataType::NDDR_OFFSET,
            CkptDataType::TABLE_2_THRESH
        };
        const set<CkptDataType> int64TransSet{
            CkptDataType::EMB_HASHMAP,
            CkptDataType::DEV_OFFSET,
            CkptDataType::HIST_REC,
            CkptDataType::NDDR_FEATMAP,
            CkptDataType::DDR_FREQ_MAP,
            CkptDataType::EXCLUDE_FREQ_MAP,
            CkptDataType::KEY_COUNT_MAP,
            CkptDataType::EVICT_POS
        };

        vector<unique_ptr<CkptDataHandler>> dataHandlers;
        string processPath;
        string innerDirPath;

        int rankId;
        int deviceId;
        int rankSize;
        bool useDynamicExpansion {false};
        vector<EmbInfo> mgmtEmbInfo;

        unique_ptr<FileSystem> fileSystemPtr;

        const int embHashNum { 2 };
        const int attribEmbDataOuterIdx { 0 };
        const int attribEmbDataInnerIdx { 1 };
        const int keyAddrElem { 2 };

        const uint32_t fourBytes = 4;

        void SetDataHandler(CkptData& ckptData);
        void SetDataHandler(const vector<CkptFeatureType>& featureTypes);

        void SaveProcess(CkptData& ckptData);
        void MakeUpperLayerSaveDir();
        void MakeDataLayerSaveDir(const vector<string>& embNames, const vector<CkptDataType>& saveDataTypes,
            const unique_ptr<CkptDataHandler>& dataHandler);
        void MakeSaveDir(const string& dirName) const;
        void SaveDataset(const vector<string>& embNames, const vector<CkptDataType>& saveDataTypes,
            const unique_ptr<CkptDataHandler>& dataHandler);
        void WriteStream(CkptTransData& transData, const string& dataDir, size_t dataSize, CkptDataType dataType);

        struct EmbSizeInfo {
            int embSize = 0;
            int extEmbSize = 0;  // embSize + (optimizer's slot) * embSize
        };
        EmbSizeInfo GetEmbeddingSize(const string& embName);
        bool CheckEmbNames(const string& embName);

        void LoadProcess(CkptData& ckptData);
        vector<string> GetEmbedTableNames();
        vector<string> GetTableLayerLoadDir();
        void LoadDataset(const vector<string>& embNames, const vector<CkptDataType>& saveDataTypes,
            const unique_ptr<CkptDataHandler>& dataHandler, CkptData& ckptData);
        void ReadStream(CkptTransData& transData, const string& dataDir, CkptDataType dataType, uint32_t dataElmtBytes);

        void ReadStreamForEmbData(CkptTransData& transData, const string& dataDir, uint32_t dataElmtBytes,
                                  CkptData& ckptData, string embName) const;
        void SetTransDataSize(CkptTransData& transData, size_t datasetSize, CkptDataType dataType);
        void CheckFileSystemPtr() const;
    };
}

#endif // MX_REC_CHECKPOINT_H
