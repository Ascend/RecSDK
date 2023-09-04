/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2023-08-17
 */

#ifndef MX_REC_KEY_FREQ_MAP_CKPT_H
#define MX_REC_KEY_FREQ_MAP_CKPT_H

#include "ckpt_data_handler/ckpt_data_handler.h"

namespace MxRec {
    using namespace std;

    class KeyFreqMapCkpt : public CkptDataHandler {
    public:
        KeyFreqMapCkpt() = default;
        ~KeyFreqMapCkpt() override {}

        void SetProcessData(CkptData& processData) override;
        void GetProcessData(CkptData& processData) override;

        vector<CkptDataType> GetDataTypes() override;

        vector<string> GetDirNames() override;
        vector<string> GetEmbNames() override;
        CkptTransData GetDataset(CkptDataType dataType, string embName) override;

        void SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData) override;

    private:
        const vector<string> fileDirNames { "HashTable", "SSD" };
        const vector<CkptDataType> saveDataTypes { CkptDataType::DDR_FREQ_MAP, CkptDataType::EXCLUDE_FREQ_MAP};

        const int freqMapElmtNum { 2 }; // Number of element types in the keyFreqMap during saving

        key_freq_mem_t saveDDRKeyFreqMaps;
        key_freq_mem_t loadDDRKeyFreqMaps;
        key_freq_mem_t saveExcludeDDRKeyFreqMaps;
        key_freq_mem_t loadExcludeDDRKeyFreqMaps;

        void SetDDRFreqMapTrans(string embName);
        void SetExcludeDDRFreqMapTrans(string embName);

        void SetDDRFreqMaps(string embName);
        void SetExcludeDDRFreqMaps(string embName);
    };
}

#endif // MX_REC_KEY_FREQ_MAP_CKPT_H
