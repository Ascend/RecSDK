#include "register/tilingdata_base.h"

namespace optiling
{
    BEGIN_TILING_DATA_DEF(TilingData1)
    TILING_DATA_FIELD_DEF(int32_t, update_dim);
    TILING_DATA_FIELD_DEF(int32_t, addr_nums);
    TILING_DATA_FIELD_DEF(int32_t, ub_limit);
    TILING_DATA_FIELD_DEF(int32_t, embbeding_type);
    TILING_DATA_FIELD_DEF(int32_t, update_type);
    END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(EmbeddingLookupByAddress, TilingData1)
}