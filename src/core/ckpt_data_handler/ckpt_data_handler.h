/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-10
 */

#ifndef MX_REC_CKPT_DATA_HANDLER_H
#define MX_REC_CKPT_DATA_HANDLER_H

#include <functional>

#include "emb_hashmap/emb_hashmap.h"
#include "host_emb/host_emb.h"
#include "utils/common.h"

namespace MxRec {
    using namespace std;

    class CkptDataHandler {
    public:
        CkptDataHandler() = default;
        virtual ~CkptDataHandler() {};

        virtual void SetProcessData(CkptData& processData) = 0;
        virtual void GetProcessData(CkptData& processData) = 0;

        virtual vector<CkptDataType> GetDataTypes() = 0;
        uint32_t GetDataElmtBytes(CkptDataType dataType);
        string GetDataDirName(CkptDataType dataType);

        virtual vector<string> GetDirNames() = 0;
        virtual vector<string> GetEmbNames() = 0;
        virtual CkptTransData GetDataset(CkptDataType dataType, string embName) = 0;

        virtual void SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData) = 0;

        virtual void SetDatasetForLoadEmb(
                CkptDataType dataType, string embName, CkptTransData& loadedData, CkptData& ckptData);

    protected:
        const vector<string> dataDirNames {
            "embedding_info",
            "embedding_data",
            "embedding_hashmap",
            "dev_offset_2_Batch_n_Key",
            "embedding_current_status",
            "max_offset",
            "key_offset_map",
            "table_2_threshold",
            "history_record",
            "attribute",
            "ddr_key_freq_map",
            "exclude_ddr_key_freq_map",
            "evict_pos"
        };
        const vector<uint32_t> dataElmtBytes { 4, 4, 8, 8, 4, 4, 8, 4, 8, 8, 8, 8, 8};

        const uint32_t eightBytes { 8 };
        const uint32_t fourBytes { 4 };

        CkptTransData transferData;

        void CleanTransfer();
    };
}

#endif // MX_REC_CKPT_DATA_HANDLER_H
