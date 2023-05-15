/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description:
 * Author: MindX SDK
 * Create: 2022-11-14
 */

#ifndef MX_REC_EMB_HASH_CKPT_H
#define MX_REC_EMB_HASH_CKPT_H

#include "ckpt_data_handler/ckpt_data_handler.h"

namespace MxRec {
    using namespace std;

    class EmbHashCkpt : public CkptDataHandler {
    public:
        EmbHashCkpt() = default;
        ~EmbHashCkpt() override {}

        void SetProcessData(CkptData& processData) override;
        void GetProcessData(CkptData& processData) override;

        vector<CkptDataType> GetDataTypes() override;

        vector<string> GetDirNames() override;
        vector<string> GetEmbNames() override;
        CkptTransData GetDataset(CkptDataType dataType, string embName) override;

        void SetDataset(CkptDataType dataType, string embName, CkptTransData& loadedData) override;

    private:
        const vector<string> fileDirNames { "HashTable", "DDR" };
        const vector<CkptDataType> saveDataTypes { CkptDataType::EMB_HASHMAP, CkptDataType::DEV_OFFSET,
            CkptDataType::EMB_CURR_STAT };

        const int currUpdataPosIdx { 0 };
        const int hostVocabIdx { 1 };
        const int devVocabIdx { 2 };

        const int attrbDev2BatchIdx { 0 };
        const int attrbDev2KeyIdx { 1 };

        const int embHashElmtNum { 2 };
        const int embCurrStatNum { 3 };
        emb_hash_mem_t saveEmbHashMaps;
        emb_hash_mem_t loadEmbHashMaps;

        void SetEmbHashMapTrans(string embName);
        void SetDevOffsetTrans(string embName);
        void SetEmbCurrStatTrans(string embName);

        void SetEmbHashMap(string embName);
        void SetDevOffset(string embName);
        void SetEmbCurrStat(string embName);
    };
}

#endif // MX_REC_EMB_HASH_CKPT_H