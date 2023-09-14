#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd

import logging.config
import os
import yaml

from mx_rec.constants.constants import LOG_MAX_SIZE
from mx_rec.validator.validator import FileValidator
from mx_rec.util.global_env_conf import global_env


def init_sys_log():
    work_dir = os.path.dirname(os.path.dirname(__file__))
    log_cfg_file = os.path.join(work_dir, "logger.yaml")
    real_config_path = os.path.realpath(log_cfg_file)

    if not FileValidator("log_cfg_file", log_cfg_file).check_file_size(real_config_path).check().is_valid():
        raise ValueError("Config file size is not valid.")

    with open(real_config_path, 'r', encoding='utf-8') as open_file:
        if not FileValidator("log_cfg_file", real_config_path). \
                check_file_size(LOG_MAX_SIZE). \
                check_not_soft_link(). \
                check_user_group(). \
                is_valid():
            raise ValueError("Log config file is not valid.")

        data = open_file.read(LOG_MAX_SIZE)
        log_cfg = yaml.safe_load(data)

    logging.config.dictConfig(log_cfg)


init_sys_log()
srv_stream_log = logging.getLogger("logStream")
srv_log = srv_stream_log
srv_log.setLevel(global_env.mxrec_log_level)
