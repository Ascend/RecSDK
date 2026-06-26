#!/bin/bash
set -e

echo "----------------        install torchrec_npu        ----------------"
cd /workspace/ && ls -hl && ls -1 Ascend-mindxsdk-*1.2.0*.tar.gz | xargs -I {} tar -zxvf {}
pip3 install *$(uname -m)*.whl --no-deps --force-reinstall

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
    # Run from a writable work dir: the source tree (test_dir) is typically root-owned
    # and not writable, which makes the relative "save_dir" save path fail permission
    # checks. Local sibling modules (dataset/model/util) are kept importable via PYTHONPATH.
    WORK_DIR=$(mktemp -d)
    cd $WORK_DIR
    PYTHONPATH=$test_dir:$PYTHONPATH pytest -x $PROJECT_DIR/$test_file
    rm -rf $WORK_DIR
done
