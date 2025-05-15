#!/bin/bash
local_rank_size=$1
host=localhost
py=$2
my_dim=$3
chongfudu=$4
all2all=$5
pre=$6
slp=$7
rm -rf /root/atc_data/*
rm -rf /root/ascend/*
rm -rf kernel_meta_*


export ALL2ALL=$5
export HOST_PIPELINE_OPS_LIB_PATH=/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc/libasc_ops.so
export EMPTY_TENSOR=1
export ENABLE_RUNTIME_V2=0
mpi_path=/usr/local/openmpi/bin/
so_path=/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc/
interface="enp61s0f0"
ulimit -c 0
export ASCEND_GLOBAL_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=3
export ASCEND_INSTALL_PATH=/usr/local/Ascend/latest/
export ASCEND_HOME_PATH=${ASCEND_INSTALL_PATH}
export ASCEND_LATEST_INSTALL_PATH=/usr/local/Ascend
CANN_BIN_PATH=${ASCEND_HOME_PATH}/bin:${ASCEND_HOME_PATH}/compiler/ccec_compiler/bin
CANN_PYTHONPATH=${ASCEND_HOME_PATH}/python/site-packages:${ASCEND_HOME_PATH}/opp/op_impl/built-in/ai_core/tbe #:${ASCEND_INSTALL_PATH}/tfplugin/latest/python/site-packages
PYTHON_BIN_PATH=/usr/local/python3.7.5/bin/
export PATH=${mpi_path}/bin:${PYTHON_BIN_PATH}:${CANN_BIN_PATH}:$PATH
export PYTHONPATH=${PYTHONPATH}:/usr/local/Ascend/latest/python/site-packages:${so_path}:${CANN_PYTHONPATH}
export LD_PRELOAD=/lib64/libgomp.so.1
CANN_LD_PATH=${ASCEND_HOME_PATH}/runtime/lib64:${ASCEND_HOME_PATH}/fwkacllib/lib64:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/lib64/plugin/opskernel:${ASCEND_HOME_PATH}/lib64/plugin/nnengine
export LD_LIBRARY_PATH=${so_path}:/usr/local/python3.7.5/lib/python3.7/site-packages/mx_rec/libasc/:/home/insert/src/platform/securec/lib/:${CANN_LD_PATH}:/home/opensource/opensource/hdf5/lib:/usr/local/lib:/usr/local/python3.7.5/lib:$LD_LIBRARY_PATH
export ASCEND_AICPU_PATH=${ASCEND_HOME_PATH}
export ASCEND_OPP_PATH=${ASCEND_HOME_PATH}/opp
export TOOLCHAIN_HOME=${ASCEND_HOME_PATH}/toolkit

export BETTER_EXCEPTIONS=1
mpi_args='-x BIND_INFO="0:48 48:48 96:48" -x SPDLOG_LEVEL=debug -bind-to none'
# rm logs
rm *txt >/dev/null
rm -rf /root/ascend/log/*

# rm shm
for i in $(ipcs -m | tail -n +4 | awk {'print $2'}); do
  ipcrm -m $i
done

num_process=${local_rank_size}
host_string=${host//_/:${local_rank_size},node}:${local_rank_size}
echo run in $host_string

interface="lo"

#python3.7 -c "import tensorflow;print(tensorflow.__path__)"
horovodrun --network-interface ${interface} -np ${num_process} --mpi-args "${mpi_args}" --mpi -H localhost:${local_rank_size} \
  python3.7 ${py} --local_rank_size ${local_rank_size} --hccl_json  hccl_json_${local_rank_size}p.json --my_dim ${my_dim} --chongfudu $chongfudu  --pre $pre --slp $slp |tee temp_{$my_dim}_{$chongfudu}_{$ALL2ALL}_{$pre}_{$slp}.log
