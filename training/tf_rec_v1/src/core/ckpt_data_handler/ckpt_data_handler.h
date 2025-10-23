/* Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

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

#ifndef MX_REC_CKPT_DATA_HANDLER_H
#define MX_REC_CKPT_DATA_HANDLER_H

#include <functional>

#include "utils/common.h"
#include "error/error.h"

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
         // The dataTypeInfoMap stores the directory name and number of bytes of storage elements
         // corresponding to each data type.
        const map<CkptDataType, pair<string, uint32_t>> dataTypeInfoMap {
                {CkptDataType::EMB_INFO,  make_pair("embedding_info", 4) },
                {CkptDataType::EMB_DATA, make_pair("embedding_data", 4) },
                {CkptDataType::EMB_HASHMAP, make_pair("embedding_hashmap", 8) },
                {CkptDataType::DEV_OFFSET, make_pair("dev_offset_2_Batch_n_Key", 8) },
                {CkptDataType::EMB_CURR_STAT, make_pair("embedding_current_status", 4)},
                {CkptDataType::NDDR_OFFSET, make_pair("max_offset", 4)},
                {CkptDataType::NDDR_FEATMAP, make_pair("key", 8)},
                {CkptDataType::TABLE_2_THRESH, make_pair("table_2_threshold", 4)},
                {CkptDataType::HIST_REC, make_pair("history_record", 8)},
                {CkptDataType::ATTRIBUTE, make_pair("attribute", 8)},
                {CkptDataType::DDR_FREQ_MAP, make_pair("ddr_key_freq_map", 8)},
                {CkptDataType::EXCLUDE_FREQ_MAP, make_pair("exclude_ddr_key_freq_map", 8)},
                {CkptDataType::EVICT_POS, make_pair("evict_pos", 8)},
                {CkptDataType::KEY_COUNT_MAP, make_pair("key_count_map", 8)}
        };

        const uint32_t eightBytes = 8;
        const uint32_t fourBytes = 4;

        CkptTransData transferData;

        void CleanTransfer();
    };
}

#endif // MX_REC_CKPT_DATA_HANDLER_H
