#define HEAD_NUM_SWITCH(COND, CONST_NAME, ...)   \
    if (COND == 4) {                             \
        constexpr static int64_t CONST_NAME = 4; \
        __VA_ARGS__;                             \
    } else if (COND == 8) {                      \
        constexpr static int64_t CONST_NAME = 8; \
        __VA_ARGS__;                             \
    }

#define HEAD_DIM_SWITCH(COND, CONST_NAME, ...)     \
    if (COND == 64) {                              \
        constexpr static int64_t CONST_NAME = 64;  \
        __VA_ARGS__;                               \
    } else if (COND == 128) {                      \
        constexpr static int64_t CONST_NAME = 128; \
        __VA_ARGS__;                               \
    }