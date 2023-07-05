/*
 * @Copyright: Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * @Description:
 * @Version: 1.0
 * @Author: dev
 * @Date: 2023-05-5 09:50:00
 * @LastEditors: dev
 * @LastEditTime: 2023-05-5 09:50:00
 */

#ifndef OCK_UNIQUE_H
#define OCK_UNIQUE_H
#include <vector>
#include <future>

#ifdef __cplusplus
extern "C" {
#endif

using ExternalThread = void (*)(const std::vector<std::function<void()>> &tasks);

#ifdef __cplusplus
}
#endif

namespace ock {
namespace ctr {
using BucketStrategy = enum class BucketStrategy {
    MODULO
};

using DataType = enum class DataType {
    INT64 = 0,
    INT32
};

using OutputType = enum class OutputType {
    NORMAL = 0,
    ENHANCED
};

using UniqueConf = struct UniqueConfCTR {
    BucketStrategy bucketStrategy = BucketStrategy::MODULO;
    OutputType outputType = OutputType::NORMAL; // 是否为普通unique
    DataType dataType = DataType::INT64;        // 输入id类型
    bool usePadding = false;                    // 是否开启padding, 开启前要先开启sharding
    bool useIdCount = false;                    // 是否开启id计数
    bool useSharding = false;                   // 是否开启sharding
    int shardingNum = -1;                       // 分桶个数
    uint32_t desiredSize = 256;                 // 预估id所需内存空间
    int paddingSize = 0;                        // 填充后长度
    int paddingVal = -1;                        // 填充值
    uint32_t minThreadNum = 1;                  // 最小工作线程数
    uint32_t maxThreadNum = 8;                  // 最大工作线程数
    int64_t maxIdVal = 0;                       // 最大id值
    bool trace = false;                         // 是否开启性能检测，需要配合外部日志输出
} __attribute__((packed));

using UniqueIn = struct UniqueInCTR {
    void *inputId = nullptr; // 输入的ids首地址(需要用户申请)必填
    uint32_t inputIdCnt = 0; // 输入ids的个数
};

using UniqueOut = struct UniqueOutCTR {
    void *uniqueId = nullptr;  // 去重分桶填充之后最终的的ids(需要用户申请)必选
    uint32_t *index = nullptr; // 去重后id的索引位置(需要用户申请)必选
    int uniqueIdCnt = 0;       // 去重后的id个数
};

using EnhancedUniqueOut = struct EnhancedUniqueOutCTR {
    void *uniqueId = nullptr;         // 去重分桶填充之后最终的的ids(需要用户申请)必选
    uint32_t *index = nullptr;        // 去重后id的索引位置(需要用户申请)必选
    void *uniqueIdInBucket = nullptr; // 去重之后的分桶内的ids(需要用户申请) sharding开启之后必须申请
    int *uniqueIdCntInBucket = nullptr; // 每个桶去重后的id个数(需要用户申请) sharding开启之后必须申请
    int uniqueIdCnt = 0;                // 去重后的id个数
    int *idCnt = nullptr;               // 每个id的重复次数(需要用户申请) 开启idCnt之后必选
    int *idCntFill = nullptr; // 每个id的重复次数带了填充(需要用户申请) 开启idCnt和padding之后必选
};

class Unique {
public:
    virtual ~Unique() = default;
    /* *
     * 初始化unique 所需配置项
     *
     * @param conf 输入unique所需的配置
     * @return error_code
     */
    virtual int Initialize(const UniqueConf &conf) = 0;

    /* *
     * 释放unique资源
     */
    virtual void UnInitialize() = 0;

    /* *
     * id去重接口
     *
     * @param UniqueIn 入参:unique用户输入
     * @param UniqueOut 出参:unique用户输出
     * @return errorCode
     */
    virtual int DoUnique(UniqueIn &uniqueIn, UniqueOut &uniqueOut) = 0;

    /* *
     * 具有额外输出的unique
     *
     * @param uniqueIn
     * @param EnhancedUniqueOut
     * @return errorCode
     */
    virtual int DoEnhancedUnique(UniqueIn &uniqueIn, EnhancedUniqueOut &enhancedUniqueOut) = 0;

    /* *
     * 设置外部线程池方法
     *
     * @return
     */
    virtual int SetExternalThreadFuncInner(ExternalThread threadFunc) = 0;
};
}
}

#endif // OCK_UNIQUE_H
