"""
配置文件
"""
from __future__ import absolute_import
import os
import json
import psutil


def get_path():
    """
    打印当前行号
    """
    return os.path.dirname(__file__)


def gen_config(server_str, local_rank_size, path=None):
    """
    生成hccl配置
    """

    def _device(local_rank_id, rank_id, server_id):
        return {
            "device_id": f"{local_rank_id}",
            "device_ip": f'192.{local_rank_id % 4}.{server_id}.{1 + local_rank_id // 4}',
            "rank_id": f"{rank_id}"
        }

    def _server(server_id):
        return {
            "device": [],
            "server_id": f"90.91.141.{server_id}"
        }

    conf = {
        "server_count": "-1",
        "server_list": [],
        "status": "completed",
        "version": "1.0"
    }
    rank_id = 0
    servers = str(server_str).split('_')
    conf['server_count'] = str(len(servers))
    for server in servers:
        srv = _server(server)
        for local_rank_id in range(local_rank_size):
            dev = _device(local_rank_id, rank_id, server)
            rank_id = rank_id + 1
            srv["device"].append(dev)
        conf['server_list'].append(srv)

    conf_str = json.dumps(conf)
    if path is None:
        path = '/tmp/hccl.json'
    with open(path, 'w') as file_handle:
        file_handle.write(conf_str)


def set_ascend_env(rank, rank_size, local_rank_size, host, file=None, dev_id=-1, dev_index=-1):
    """
    配置昇腾相关的参数和环境变量，生成hccl配置
    """
    rank = str(rank)
    rank_size = str(rank_size)
    local_rank_size = int(local_rank_size)
    host = str(host)

    os.environ["MOX_USE_NPU"] = "1"
    os.environ["FUSION_TENSOR_SIZE"] = "2000000000"
    os.environ["MOX_USE_TF_ESTIMATOR"] = "0"
    os.environ["MOX_USE_TDT"] = "1"
    os.environ["HEARTBEAT"] = "1"
    os.environ["CONITNUE_TRAIN"] = "true"

    os.environ["RANK_ID"] = rank
    local_rank_id = int(rank) % int(local_rank_size)
    if dev_id != -1:
        os.environ["DEVICE_ID"] = str(dev_id)
        os.environ["ASCEND_DEVICE_ID"] = str(dev_id)
    else:
        os.environ["DEVICE_ID"] = str(local_rank_id)
        os.environ["ASCEND_DEVICE_ID"] = str(local_rank_id)
    if dev_index != -1:
        os.environ["DEVICE_INDEX"] = str(dev_index)
    else:
        os.environ["DEVICE_INDEX"] = str(local_rank_id)

    os.environ["RANK_SIZE"] = rank_size
    if file:
        os.environ["RANK_TABLE_FILE"] = file
    else:
        gen_config(host, local_rank_size)
        os.environ["RANK_TABLE_FILE"] = "/tmp/hccl.json"
    os.environ["HCCL_CONNECT_TIMEOUT"] = "600"

    os.environ["JOB_ID"] = "10086"
    os.environ["SOC_VERSION"] = "Ascend910"
    os.environ["GE_AICPU_FLAG"] = "1"
    os.environ["NEW_GE_FE_ID"] = "1"
    os.environ["EXPERIMENTAL_DYNAMIC_PARTITION"] = "1"
    os.environ["ENABLE_FORCE_V2_CONTROL"] = "1"


def bind_cpu():
    p = psutil.Process()
    try:
        bind_start = 48
        bind_count = 96
        p.cpu_affinity([bind_start + x for x in range(bind_count)])
    except IndexError:
        print("error cpu bind info, skipped.")
