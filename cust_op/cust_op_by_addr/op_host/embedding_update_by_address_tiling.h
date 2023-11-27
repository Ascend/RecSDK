#ifndef EMBEDDING_UPDATE_BY_ADDRESS_TILING_H
#define EMBEDDING_UPDATE_BY_ADDRESS_TILING_H
#include "register/tilingdata_base.h"

namespace optiling
{
BEGIN_TILING_DATA_DEF(TilingData2)
    TILING_DATA_FIELD_DEF(int32_t, update_dim);
    TILING_DATA_FIELD_DEF(int32_t, addr_nums);
    TILING_DATA_FIELD_DEF(int32_t, embedding_type);
    TILING_DATA_FIELD_DEF(int32_t, update_type);
    TILING_DATA_FIELD_DEF(int32_t, addr_per_loop);
    TILING_DATA_FIELD_DEF(int32_t, ping_pong_num);
    TILING_DATA_FIELD_DEF(int32_t, type_size);
    TILING_DATA_FIELD_DEF(int32_t, input_dim_aligned);

END_TILING_DATA_DEF;

    REGISTER_TILING_DATA_CLASS(EmbeddingUpdateByAddress, TilingData2)
}
#endif // EMBEDDING_UPDATE_BY_ADDRESS_TILING_H