#!/usr/bin/env python3
# coding: UTF-8
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
from datetime import datetime, timezone
import logging
import os
import stat
import time
import threading


CURRENT_PATH = os.getcwd()
FORMATTED_TIME = datetime.now(timezone.utc).strftime("%d_%H_%M_%S")
TRAIN_LOG_PATH = f"{CURRENT_PATH}/train_{FORMATTED_TIME}.log"
EVAL_LOG_PATH = f"{CURRENT_PATH}/eval_{FORMATTED_TIME}.log"
STAT_PREFIX = "[StatInfo]"
DISPLAY_MODE_PRINT_SCREEN = "print_screen"
DISPLAY_MODE_SAVE_LOG = "save_log"
TABLE_NUM_LINE_PREFIX = "current_table_num"
CHANNEL_LINE_PREFIX = "channel_id"
VALUE_READ_START = 6
VALUE_READ_INTERVAL = 2
LOOP_SLEEP_TIME = 0.003

# run mode to be selected
HBM_NORMAL = {"key_process_time_cost": "key_process_time_cost",
                   "batch_key_num": "batch_key_num",
                   "unique_key_num": "unique_key_num"}
HBM_FAAE = {"key_process_time_cost": "key_process_time_cost",
                 "batch_key_num": "batch_key_num",
                 "unique_key_num": "faae_unique_key_num"}
HBM_HOT = {"key_process_time_cost": "key_process_time_cost",
                "batch_key_num": "batch_key_num",
                "unique_key_num": "hot_unique_key_num"}
HBM_FAST = {"key_process_time_cost": "key_process_time_cost_with_fast_unique",
                 "batch_key_num": "batch_key_num_with_fast_unique",
                 "unique_key_num": "unique_key_num_with_fast_unique"}

DDR_NORMAL = {"key_process_time_cost": "key_process_time_cost",
                   "batch_key_num": "batch_key_num",
                   "unique_key_num": "unique_key_num",
                   "swap_key_size": "swap_key_size",
                   "swap_time_cost": "swap_time_cost"}
DDR_FAAE = {"key_process_time_cost": "key_process_time_cost",
                 "batch_key_num": "batch_key_num",
                 "unique_key_num": "faae_unique_key_num",
                 "swap_key_size": "swap_key_size",
                 "swap_time_cost": "swap_time_cost"}
DDR_HOT = {"key_process_time_cost": "key_process_time_cost",
                "batch_key_num": "batch_key_num",
                "unique_key_num": "hot_unique_key_num",
                "swap_key_size": "swap_key_size",
                "swap_time_cost": "swap_time_cost"}
DDR_FAST = {"key_process_time_cost": "key_process_time_cost_with_fast_unique",
                 "batch_key_num": "batch_key_num_with_fast_unique",
                 "unique_key_num": "unique_key_num_with_fast_unique",
                 "swap_key_size": "swap_key_size",
                 "swap_time_cost": "swap_time_cost"}

DDR_LIST = [DDR_NORMAL, DDR_FAAE, DDR_HOT, DDR_FAST]

# ======================  Please modify here according to readme before using ======================
TARGET_REC_LOG_PATH = "/home/example.log"
RUN_MODE = DDR_FAST
RANK_SIZE = 8
DISPLAY_MODE = "save_log"  # can be "save_log" or "print_screen"
DISPLAY_INTERVAL = 1
# ==================================================================================================

TRAIN_DICT = dict()
EVAL_DICT = dict()
CURRENT_TABLE_NUM = 0

FULL_DICT_LEN = len(RUN_MODE)
TRAIN_TOTAL_DATA = dict()
EVAL_TOTAL_DATA = dict()


def read_log_by_line_loop(log_add: str):
    """
    read log by line continuously

    Arg:
        log_add: file address to read mxRec log file
    """
    logging.info("=============  log reading started  =============")
    with open(log_add, 'r') as log_file:
        while True:
            new_line = log_file.readline()  # 读取一行新增内容
            check_line_content(new_line)
    return


def check_line_content(line):
    """
    check line file content and record relevant info

    Arg:
        line: line object from file reading
    """
    index = line.find("[StatInfo]")
    if line and index != -1:
        stat_data = line[index + len(STAT_PREFIX):].split()
        if stat_data[0] == CHANNEL_LINE_PREFIX:
            tar_dict = create_data(stat_data)
            update_data(stat_data, tar_dict)
        elif stat_data[0] == TABLE_NUM_LINE_PREFIX:
            global CURRENT_TABLE_NUM
            CURRENT_TABLE_NUM = int(stat_data[1])
    else:
        # 没有新增内容时，可以选择休眠一段时间再继续读取，避免过度消耗资源
        time.sleep(LOOP_SLEEP_TIME)
    return


def start_display_data(channel: int):
    """
    start to display data continuously

    Arg:
        channel: channel id 0 while train, 1 while eval
    """
    logging.info("=============  channel: %d stat display  =============", channel)
    if channel == 0:
        glob_dict = TRAIN_DICT
    elif channel == 1:
        glob_dict = EVAL_DICT
    else:
        raise ValueError("channel num can only be 0 or 1")
    display_per_step(glob_dict, channel)
    return


def display_per_step(glob_dict: dict, channel: int):
    """
    display stat info according to step num

    Arg:
        glob_dict: which dict to use to record stat info, can be TRAIN_DICT or EVAL_DICT
        channel: channel id 0 while train, 1 while eval
    """
    step = 0
    while True:
        if step not in glob_dict:
            time.sleep(LOOP_SLEEP_TIME)
            continue
        display_per_rank(glob_dict[step], channel, step)
        del glob_dict[step]
        step += 1
    return


def display_per_rank(current_step_dict: dict, channel: int, step: int):
    """
    display stat info in each step according to rank id

    Arg:
        channel: channel id 0 while train, 1 while eval
        step: current step num
    """
    i = 0
    while i < RANK_SIZE:
        if i not in current_step_dict:
            time.sleep(LOOP_SLEEP_TIME)
            continue
        if len(current_step_dict[i]) == FULL_DICT_LEN:
            display_data(current_step_dict[i], channel, step, i)
            i += 1
        elif len(current_step_dict[i]) < FULL_DICT_LEN:
            time.sleep(LOOP_SLEEP_TIME)
        else:
            raise ValueError("dict length shall not be bigger than FULL_DICT_LEN")
    return


def create_total_dict():
    """
    create dict instance according to template
    """
    template_dict = {
        "total_batch_key_num": 0,
        "total_unique_key_num": 0,
        "total_key_process_time_cost": 0,
        "total_swap_size": 0,
        "total_swap_time": 0
    }
    return template_dict.copy()


def construct_ddr_message(display_dict: dict, target_dict: dict, batch_key_num: int, total_batch_key_num: int):
    """
    construct ddr info message to display

    Arg:
        display_dict: info dict to display
        target_dict: which total dict to update stat info
        batch_key_num: key num of current batch in current device
        total_batch_key_num: total key num in current device
    """
    swap_key_size = display_dict[RUN_MODE["swap_key_size"]]
    target_dict["total_swap_size"] += swap_key_size
    total_swap_size = target_dict["total_swap_size"]

    swap_time_cost = display_dict[RUN_MODE["swap_time_cost"]]
    target_dict["total_swap_time"] += swap_time_cost
    total_swap_time = target_dict["total_swap_time"]

    swap_speed = 0
    if swap_time_cost != 0:
        swap_speed = swap_key_size / swap_time_cost

    total_swap_speed = 0
    if total_swap_time != 0:
        total_swap_speed = total_swap_size / total_swap_time

    ddr_message = f"Current Swap Key Num:{swap_key_size}  " \
                  f"\nCurrent Swap Speed:{round(swap_speed, 3)}" \
                  f"\nCurrent HBM Rate:{round(((batch_key_num - swap_key_size) / batch_key_num), 3)}" \
                  f"\nToTal Swap Key Num:{total_swap_size} " \
                  f"\nAverage Swap Speed:{round(total_swap_speed, 3)}" \
                  f"\nAverage HBM Rate:{round(((total_batch_key_num - total_swap_size) / total_batch_key_num), 3)}\n"
    return ddr_message


def display_data(display_dict: dict, channel: int, step: int, rank_id: int):
    """
    display stat info messages according to DISPLAY_MODE

    Arg:
        display_dict: info dict to display
        channel: channel id 0 while train, 1 while eval
        step: current step num
        rank_id: id of current device rank
    """
    if channel == 0:
        target_dict = TRAIN_TOTAL_DATA
    else:
        target_dict = EVAL_TOTAL_DATA
    if rank_id not in target_dict:
        target_dict[rank_id] = create_total_dict()
    target_dict = target_dict[rank_id]
    batch_key_num = display_dict[RUN_MODE["batch_key_num"]]
    target_dict["total_batch_key_num"] += batch_key_num
    total_batch_key_num = target_dict["total_batch_key_num"]

    unique_key_num = display_dict[RUN_MODE["unique_key_num"]]
    target_dict["total_unique_key_num"] += unique_key_num
    total_unique_key_num = target_dict["total_unique_key_num"]

    key_process_time_cost = display_dict[RUN_MODE["key_process_time_cost"]]
    target_dict["total_key_process_time_cost"] += key_process_time_cost
    total_key_process_time_cost = target_dict["total_key_process_time_cost"]

    key_process_speed = 0
    if key_process_time_cost != 0:
        key_process_speed = batch_key_num / key_process_time_cost

    total_key_process_speed = 0
    if total_key_process_time_cost != 0:
        total_key_process_speed = total_batch_key_num / total_key_process_time_cost

    message = f"[STATINFO]Channel:{channel} Current Step:{step} RankId:{rank_id} " \
              f"\nCurrentTableNum:{CURRENT_TABLE_NUM}" \
              f"\nCurrent Batch Key Num:{batch_key_num} Current Unique Key Num:{unique_key_num}" \
              f"\nCurrent Deduplication Key Rate:{round((1 - unique_key_num / batch_key_num), 3)}" \
              f"\nCurrent Key Process Speed:{round(key_process_speed, 3)} / ms" \
              f"\nToTal Batch Key Num:{total_batch_key_num} ToTal Unique Key Num:{total_unique_key_num}" \
              f"\nAverage Deduplication Key Rate: {round((1 - total_unique_key_num / total_batch_key_num), 3)}" \
              f"\nAverage Key Process Speed:{round(total_key_process_speed, 3)} / ms\n"

    if RUN_MODE in DDR_LIST:
        ddr_message = construct_ddr_message(display_dict, target_dict, batch_key_num, total_batch_key_num)
        message = message + ddr_message

    if step % DISPLAY_INTERVAL == 0:
        if DISPLAY_MODE == DISPLAY_MODE_PRINT_SCREEN:
            logging.info(message)
        elif DISPLAY_MODE == DISPLAY_MODE_SAVE_LOG:
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            modes = stat.S_IWUSR | stat.S_IRUSR
            if channel == 0:
                log_path = TRAIN_LOG_PATH
            elif channel == 1:
                log_path = EVAL_LOG_PATH
            else:
                raise ValueError("channel num can only be 0 or 1")
            with os.fdopen(os.open(log_path, flags, modes), mode='a') as log_out:
                log_out.write(message + "\n")
        else:
            raise ValueError(f"DISPLAY_MODE can only be 'print_screen' or 'save_log' but '{DISPLAY_MODE}' is given")


def create_data(line_stat_data):
    """
    store stat data according to log file

    Arg:
        line_stat_data: line object from file reading
    """
    channel_id = int(line_stat_data[1])
    step_id = int(line_stat_data[3])
    rank_id = int(line_stat_data[5])
    if channel_id == 0:
        global_dict = TRAIN_DICT
    elif channel_id == 1:
        global_dict = EVAL_DICT
    else:
        raise ValueError("channel num can only be 0 or 1 and ")

    if step_id not in global_dict:
        global_dict[step_id] = dict()
    if rank_id not in global_dict[step_id]:
        global_dict[step_id][rank_id] = dict()
    target_dict = global_dict[step_id][rank_id]
    return target_dict


def update_data(line_stat_data, target_dict: dict):
    """
    update stat data according to log file

    Arg:
        line_stat_data: line object from file reading
    """
    for i in range(VALUE_READ_START, len(line_stat_data), VALUE_READ_INTERVAL):
        target_dict[line_stat_data[i]] = int(line_stat_data[i + 1])
    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    thread1 = threading.Thread(target=read_log_by_line_loop, args=(TARGET_REC_LOG_PATH,))
    thread2 = threading.Thread(target=start_display_data, args=(0,))
    thread3 = threading.Thread(target=start_display_data, args=(1,))

    # 启动线程
    thread1.start()
    thread2.start()
    thread3.start()