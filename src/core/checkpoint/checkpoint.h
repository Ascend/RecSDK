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
#include "utils/common.h"
#include "ckpt_data_handler/ckpt_data_handler.h"
#include "buffer_queue.h"

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
        const mode_t dirMode { 0500 };

        const string currDir { "." };
        const string prevDir { ".." };

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
            CkptDataType::EXCLUDE_FREQ_MAP
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
        const int keyAddrElem { 2 };

        void SetDataHandler(CkptData& ckptData);
        void SetDataHandler(const vector<CkptFeatureType>& featureTypes);

        void SaveProcess(CkptData& ckptData);
        void MakeUpperLayerSaveDir(const vector<string>& dirNames);
        void MakeDataLayerSaveDir(const vector<string>& embNames, const vector<CkptDataType>& saveDataTypes,
            const unique_ptr<CkptDataHandler>& dataHandler);
        void MakeSaveDir(const string& dirName) const;
        void SaveDataset(const vector<string>& embNames, const vector<CkptDataType>& saveDataTypes,
            const unique_ptr<CkptDataHandler>& dataHandler);
        void WriteStream(CkptTransData& transData, const string& dataDir, size_t dataSize, CkptDataType dataType);
        void FillToBuffer(BufferQueue& queue, const char* data, size_t dataSize);
        void WriteDataset(CkptTransData& transData,
                                      int fd,
                                      size_t writeSize,
                                      CkptDataType dataType,
                                      size_t idx);

        void WriterFn(BufferQueue& queue, int fd);

        void WriteEmbedding(const CkptTransData& transData, const string& dataDir, const int& embeddingSize);
        void ReadEmbedding(CkptTransData& transData, const string& dataDir, const string& embName);

        struct EmbSizeInfo {
            int embSize = 0;
            int extEmbSize = 0;  // embSize + (optimizer's slot) * embSize
        };
        EmbSizeInfo GetEmbeddingSize(const string& embName);
        bool CheckEmbNames(const string& embName);

        void LoadProcess(CkptData& ckptData);
        void GetUpperLayerLoadDir(const vector<string>& dirNames);
        vector<string> GetEmbedTableNames();
        vector<string> GetTableLayerLoadDir();
        void LoadDataset(const vector<string>& embNames, const vector<CkptDataType>& saveDataTypes,
            const unique_ptr<CkptDataHandler>& dataHandler, CkptData& ckptData);
        void ReadStream(CkptTransData& transData, const string& dataDir, CkptDataType dataType, uint32_t dataElmtBytes);
        void ValidateFile(int fd, const string& dataDir, size_t datasetSize) const;
        void HandleMappedData(char* mappedData, size_t mapRowNum, size_t onceReadByteSize,
                                          vector<vector<float>>& dst, size_t cnt) const;
        void CalculateMapSize(off_t fileSize, size_t& mapByteSize, size_t& mapRowNum, size_t onceReadByteSize) const;

        void ReadStreamForEmbData(CkptTransData& transData, const string& dataDir, uint32_t dataElmtBytes,
                                  CkptData& ckptData, string embName) const;
        void SetTransDataSize(CkptTransData& transData, size_t datasetSize, CkptDataType dataType);
        void ReadDataset(CkptTransData& transData, ifstream& readFile, size_t readSize, CkptDataType dataType,
            size_t idx);
    };
}

#endif // MX_REC_CHECKPOINT_H
