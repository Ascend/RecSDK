#define HEAD_DIM_SWITCH(PRECOND, COND, CONST_NAME, CONST_RESULT, ...)     \
    if (PRECOND && COND == 64) {                              \
        constexpr int64_t CONST_NAME = 64;  \
        constexpr bool CONST_RESULT = true; \
        __VA_ARGS__;                               \
    } else if (PRECOND && COND == 128) {                      \
        constexpr int64_t CONST_NAME = 128; \
        constexpr bool CONST_RESULT = true; \
        __VA_ARGS__;                               \
    } else if (PRECOND && COND == 256) {                      \
        constexpr int64_t CONST_NAME = 128; \
        constexpr bool CONST_RESULT = true; \
        __VA_ARGS__;                               \
    } else {                      \
        constexpr int64_t CONST_NAME = 256; \
        constexpr bool CONST_RESULT = false; \
        __VA_ARGS__;                               \
    } 