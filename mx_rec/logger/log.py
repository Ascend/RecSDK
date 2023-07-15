#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd

import logging.config
import os
import yaml

from mx_rec.constants.constants import MAX_SIZE, LOG_MAX_SIZE
from mx_rec.validator.validator import FileValidator
from mx_rec.validator.validator import DirectoryValidator


def init_sys_log():
    work_dir = os.path.dirname(os.path.dirname(__file__))
    log_cfg_file = os.path.join(work_dir, "logger.yaml")
    real_config_path = os.path.realpath(log_cfg_file)

    if not FileValidator(log_cfg_file).check_file_size(real_config_path).check().is_valid():
        raise ValueError("Config file size is not valid.")

    with open(real_config_path, 'r', encoding='utf-8') as open_file:
        if not FileValidator(real_config_path).\
                check_file_size(LOG_MAX_SIZE).\
                check_not_soft_link().\
                check_user_group().\
                is_valid():
            raise ValueError("Log config file is not valid.")

        data = open_file.read(LOG_MAX_SIZE)
        log_cfg = yaml.safe_load(data)

    logging.config.dictConfig(log_cfg)


def init_log_dir_for_dt(log_cfg):
    """Create log directory for local environment dt test.

    :param log_cfg: log configuration dictionary from yml file.
    :return: None
    """
    handlers = log_cfg.get('handlers')
    if not handlers:
        return

    for handler_name in handlers:
        handler_dict = handlers.get(handler_name)
        log_file = handler_dict.get('filename')

        if not log_file:
            continue

        log_file_standard = os.path.realpath(log_file)
        if log_file_standard != log_file:
            continue

        log_dir = os.path.dirname(log_file_standard)
        if not DirectoryValidator(log_dir) \
                .check_is_not_none() \
                .check_dir_name() \
                .should_not_contains_sensitive_words() \
                .with_blacklist() \
                .check() \
                .is_valid():
            continue


init_sys_log()
srv_stream_log = logging.getLogger("logStream")
env_log_level = os.getenv("MXREC_LOG_LEVEL")
srv_log = srv_stream_log
if env_log_level:
    srv_log.setLevel(env_log_level)

