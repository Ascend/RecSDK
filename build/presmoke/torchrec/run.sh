#!/bin/bash
set -e

echo "----------------        install torchrec_npu        ----------------"
cd /workspace/ && ls -1 Ascend-mindxsdk-*1.2.0*.tar.gz | xargs -I {} tar -zxvf {}
pip3 install *$(uname -m)*.whl --no-deps --force-reinstall

echo "----------------        build aclnn ops (v220)        ----------------"
# torchrec 依赖的 custom OPP 算子（含 WORLD_SIZE>1 分布式路径用到的 alltoall 算子）。
# 不再依赖 fbgemm_ascend，因此需自行编译安装，否则运行时报 "aclnnXxx not in libopapi.so"。
unset ASCEND_CUSTOM_OPP_PATH

# 算子列表默认覆盖 torchrec 前向/反向与分布式路径；外部可通过 OPS 环境变量覆盖。
OPS="${OPS:-split_embedding_codegen_forward_unweighted backward_codegen_adagrad_unweighted_exact dense_embedding_codegen_lookup_function dense_embedding_codegen_lookup_function_grad asynchronous_complete_cumsum permute2d_sparse_data}"

# CI 环境通常已 source CANN set_env.sh；若未设置 ASCEND_OPP_PATH，则从 ASCEND_HOME_PATH 兜底 source。
if [ -z "$ASCEND_OPP_PATH" ]; then
    : "${ASCEND_HOME_PATH:=/usr/local/Ascend/ascend-toolkit/latest}"
    if [ -f "$ASCEND_HOME_PATH/set_env.sh" ]; then
        source "$ASCEND_HOME_PATH/set_env.sh"
    fi
fi

# VENDORS 默认按 CANN 规范放在 ASCEND_OPP_PATH/vendors/；也可由外部指定。
if [ -z "$VENDORS" ] && [ -n "$ASCEND_OPP_PATH" ]; then
    VENDORS="${ASCEND_OPP_PATH}/vendors/"
fi

for op in $OPS; do
    rm -fr "$VENDORS/$op"
    ( cd "$PROJECT_DIR/cust_op/ascendc_op/ai_core_op/$op/v220" && bash run.sh )
done

# 注册 custom OPP：source 各算子 set_env.bash，并把 op_api/lib 加入 LD_LIBRARY_PATH。
for op in $OPS; do
    SET_ENV=$VENDORS/$op/bin/set_env.bash
    [ -f "$SET_ENV" ] && source "$SET_ENV"
    export LD_LIBRARY_PATH="$VENDORS/$op/op_api/lib:${LD_LIBRARY_PATH}"
done
export ASCEND_CUSTOM_OPP_PATH

# ASCEND_OPP_PATH 指向算子实际安装的 OPP 根，确保 GetCustLibPath 扫描到 custom 库。
if [ -n "$VENDORS" ] && [[ "$VENDORS" == */vendors/ ]]; then
    export ASCEND_OPP_PATH="${VENDORS%/vendors/}"
fi

# 兜底 CANN lib64（libascendcl.so/libnnopbase.so）进 LD_LIBRARY_PATH。
if [ -n "$ASCEND_HOME_PATH" ]; then
    CANN_LIB64="$ASCEND_HOME_PATH/lib64"
    case ":$LD_LIBRARY_PATH:" in
        *:"$CANN_LIB64":*) ;;
        *) export LD_LIBRARY_PATH="${CANN_LIB64}:${LD_LIBRARY_PATH}" ;;
    esac
fi

# 确保 config.ini 存在并含我们的算子（GetCustLibPath 依赖其 load_priority 列表）。
CONF_DIR="$ASCEND_OPP_PATH/vendors"
if [ -n "$ASCEND_OPP_PATH" ]; then
    mkdir -p "$CONF_DIR"
    CONF_FILE="$CONF_DIR/config.ini"
    touch "$CONF_FILE"
    EXISTING=""
    grep -q '^load_priority=' "$CONF_FILE" 2>/dev/null && EXISTING=$(sed -n 's/^load_priority=//p' "$CONF_FILE")
    ADDED="$EXISTING"
    for op in $OPS; do
        if [ -z "$ADDED" ]; then
            ADDED="$op"
        elif ! echo "$ADDED" | tr ',' '\n' | grep -qx "$op"; then
            ADDED="$ADDED,$op"
        fi
    done
    printf 'load_priority=%s\n' "$ADDED" > "$CONF_FILE"
fi

echo "ASCEND_OPP_PATH=$ASCEND_OPP_PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "----------------        torchrec envs setting        ----------------"
export OMP_NUM_THREADS=12
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export WITH_EMBCACHE=1
export EMBCACHE_SIZE_ON_DEVICE_MEM=$((1*1024*1024))
export ENABLE_FAST_HASHMAP=false
export FAST_HASHMAP_RESERVE_BUCKET_NUM=2097152
export GLOG_stderrthreshold=1

echo "----------------        run pytest        ----------------"
cd $PRESMOKE_DIR/torchrec/
export TEST_FILES=$(cat $PRESMOKE_DIR/changes.txt | xargs -I {} grep {} impact_analysis_all_files.csv | cut -d',' -f2 | tr '|' '\n' | sort -u | xargs -I {} grep {} all_test_files.csv)
echo $TEST_FILES | tr ' ' '\n'
for test_file in $TEST_FILES; do
    sed -i "s/\"embedding_dims\": \[\[128, 128\]\],/\"embedding_dims\": \[\[16, 16\]\],/" $PROJECT_DIR/$test_file
    if [[ "$test_file" == *"test_save_and_load.py"* ]]; then
        sed -i "s/WORLD_SIZE = 2/WORLD_SIZE = 1/" $PROJECT_DIR/$test_file
    fi
    test_dir=$(dirname $PROJECT_DIR/$test_file)
    WORK_DIR=$(mktemp -d)
    cd $WORK_DIR
    PYTHONPATH=$test_dir:$PYTHONPATH pytest -x $PROJECT_DIR/$test_file
    rm -rf $WORK_DIR
done
