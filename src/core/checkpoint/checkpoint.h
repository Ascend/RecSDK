/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: use to manage model saving and loading process
 * Author: MindX SDK
 * Create: 2022-11-15
 */

#ifndef MX_REC_CHECKPOINT_H
#define MX_REC_CHECKPOINT_H

#include <dirent.h>
#include <acl/acl_rt.h>
#include <acl/acl.h>

#include "ckpt_data_handler/ckpt_data_handler.h"

namespace MxRec {
    using namespace std;

    class Checkpoint {
    public:
        Checkpoint() = default;
        ~Checkpoint() {};

        void SaveModel(string savePath, CkptData& ckptData, RankInfo& mgmtRankInfo, const vector<EmbInfo>& EmbInfo);
        void LoadModel(string loadPath, CkptData& ckptData, RankInfo& mgmtRankInfo, const vector<EmbInfo>& EmbInfo,
                       const vector<CkptFeatureType>& featureTypes);

    private:
        const string datasetName { "slice_" };
        const string dataFileType { ".data" };
        const string attribFileType { ".attribute" };
        const string dirSeparator { "/" };
        const mode_t dirMode { 0777 };

        const string currDir { "." };
        const string prevDir { ".." };

        const size_t oneTimeReadWriteLen { 32768 }; // 4096 * 8

        const set<CkptDataType> int32TransSet {
            CkptDataType::EMB_INFO,
            CkptDataType::EMB_CURR_STAT,
            CkptDataType::NDDR_OFFSET,
            CkptDataType::TENSOR_2_THRESH
        };
        const set<CkptDataType> int64TransSet{
            CkptDataType::EMB_HASHMAP,
            CkptDataType::DEV_OFFSET,
            CkptDataType::HIST_REC,
            CkptDataType::NDDR_FEATMAP
        };
        const set<CkptDataType> floatTransSet{
            CkptDataType::EMB_DATA
        };

        vector<unique_ptr<CkptDataHandler>> dataHandlers;
        string processPath;
        string innerDirPath;

        int rankId;
        int deviceId;
        bool useDynamicExpansion {false};
        vector<EmbInfo> mgmtEmbInfo;

        const int embHashNum { 2 };
        const int attribEmbDataOuterIdx { 0 };
        const int attribEmbDataInnerIdx { 1 };

        void SetDataHandler(CkptData& ckptData);
        void SetDataHandler(const vector<CkptFeatureType>& featureTypes);

        void SaveProcess(CkptData& ckptData);
        void MakeUpperLayerSaveDir(const vector<string>& dirNames);
        void MakeDataLayerSaveDir(const vector<string>& embNames, const vector<CkptDataType>& saveDataTypes,
            const unique_ptr<CkptDataHandler>& dataHandler);
        void MakeSaveDir(const string& dirName);
        void SaveDataset(const vector<string>& embNames, const vector<CkptDataType>& saveDataTypes,
            const unique_ptr<CkptDataHandler>& dataHandler);
        void WriteStream(CkptTransData& transData, const string& dataDir, size_t dataSize, CkptDataType dataType);
        void WriteDataset(CkptTransData& transData, ofstream& writeFile, size_t writeSize, CkptDataType dataType,
            size_t idx);
        void WriteEmbedding(CkptTransData& transData, const string& dataDir, int& embeddingSize);
        void ReadEmbedding(CkptTransData& transData, const string& dataDir);

        int  GetEmbeddingSize(const string& embName);

        void LoadProcess(CkptData& ckptData);
        void GetUpperLayerLoadDir(const vector<string>& dirNames);
        vector<string> GetTableLayerLoadDir();
        void LoadDataset(const vector<string>& embNames, const vector<CkptDataType>& saveDataTypes,
            const unique_ptr<CkptDataHandler>& dataHandler, CkptData& ckptData);
        void ReadStream(CkptTransData& transData, const string& dataDir, CkptDataType dataType, uint32_t dataElmtBytes);
        void ReadStreamForEmbData(CkptTransData& transData, const string& dataDir, uint32_t dataElmtBytes,
                                  CkptData& ckptData, string embName);
        void SetTransDataSize(CkptTransData& transData, size_t datasetSize, CkptDataType dataType);
        void ReadDataset(CkptTransData& transData, ifstream& readFile, size_t readSize, CkptDataType dataType,
            size_t idx);
    };
}

#endif // MX_REC_CHECKPOINT_H
