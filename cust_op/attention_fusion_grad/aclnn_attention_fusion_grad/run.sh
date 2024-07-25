#!/bin/bash
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=0

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

SHORT=v:,
LONG=dtype:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        # float16, float, int32
        (-v | --dtype)
            DTYPE="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

if [ ! $ASCEND_HOME_DIR ]; then
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_DIR=$HOME/Ascend/ascend-toolkit/latest
    else
        export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

export DDK_PATH=$ASCEND_HOME_DIR
arch=$(uname -m)
export NPU_HOST_LIB=$ASCEND_HOME_DIR/${arch}-linux/lib64

function main {
    rm -rf $HOME/ascend/log/*
    rm ./input/*.bin
    rm ./output/*.bin

    cd $CURRENT_DIR
    python3 scripts/gen_data.py
    if [ $? -ne 0 ]; then
        echo "ERROR: generate input data failed!"
        return 1
    fi
    echo "INFO: generate input data success!"

    cd $CURRENT_DIR; rm -rf build; mkdir -p build; cd build
    cmake ../src
    if [ $? -ne 0 ]; then
        echo "ERROR: cmake failed!"
        return 1
    fi
    echo "INFO: cmake success!"
    make
    if [ $? -ne 0 ]; then
        echo "ERROR: make failed!"
        return 1
    fi
    echo "INFO: make success!"

    cd $CURRENT_DIR/output
    echo "INFO: execute op!"
    ./execute_attention_fusion_grad_op

    if [ $? -ne 0 ]; then
        echo "ERROR: acl executable run failed! please check your project!"
        return 1
    fi
    echo "INFO: acl executable run success!"
    cd $CURRENT_DIR
    ret=`python3 scripts/verify_result.py output/grad_query.bin output/grad_key.bin output/grad_value.bin  output/golden_grad_query.bin output/golden_grad_key.bin output/golden_grad_value.bin  `
    echo $ret
    if [ "x$ret" == "xtest pass" ]; then
        echo ""
        echo "#####################################"
        echo "INFO: you have passed the Precision!"
        echo "#####################################"
        echo ""
    fi
}

main
