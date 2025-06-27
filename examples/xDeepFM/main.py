#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This script parse and run train function"""
from npu_bridge.npu_init import *
import train
import utils.util as util
import tensorflow as tf
import sys
from utils.log import Log




def flat_config(config):
    """flat config to a dict"""
    f_config = {}
    category = ['data', 'model', 'train', 'info']
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config


def create_hparams(FLAGS):
    """Create hparams."""
    FLAGS = flat_config(FLAGS)

    hparam_specs = {
        # data
        'train_file': None,
        'eval_file': None,
        'test_file': None,
        'infer_file': None,
        'FEATURE_COUNT': None,
        'FIELD_COUNT': None,
        'data_format': None,
        'PAIR_NUM': None,
        'DNN_FIELD_NUM': None,
        'n_user': None,
        'n_item': None,
        'n_user_attr': None,
        'n_item_attr': None,

        # model
        'dim': None,
        'layer_sizes': None,
        'cross_layer_sizes': None,
        'cross_layers': None,
        'activation': None,
        'cross_activation': "identity",
        'dropout': None,
        'attention_layer_sizes': None,
        'attention_activation': None,
        'model_type': None,
        'method': None,
        'load_model_name': None,
        'mu': None,

        # train
        'init_method': 'tnormal',
        'init_value': 0.01,
        'embed_l2': 0.0000,
        'embed_l1': 0.0000,
        'layer_l2': 0.0000,
        'layer_l1': 0.0000,
        'cross_l2': 0.0000,
        'cross_l1': 0.0000,
        'learning_rate': 0.001,
        'loss': None,
        'optimizer': 'adam',
        'epochs': 10,
        'batch_size': 1,

        # show info
        'log': "log",
        'logger': None,
        'show_step': 1,
        'save_epoch': 5,
        'metrics': None,
    }

    kwargs = {key: FLAGS.get(key, default) for key, default in hparam_specs.items()}
    
    return tf.contrib.training.HParams(**kwargs)


def check_type(config):
    """check config type"""
    # check parameter type
    int_parameters = ['FEATURE_COUNT', 'FIELD_COUNT', 'dim', 'epochs', 'batch_size', 'show_step', \
                      'save_epoch', 'PAIR_NUM', 'DNN_FIELD_NUM', 'attention_layer_sizes', \
                      'n_user', 'n_item', 'n_user_attr', 'n_item_attr']
    for param in int_parameters:
        if param in config and not isinstance(config[param], int):
            raise TypeError("parameters {0} must be int".format(param))

    float_parameters = ['init_value', 'learning_rate', 'embed_l2', \
                        'embed_l1', 'layer_l2', 'layer_l1', 'mu']
    for param in float_parameters:
        if param in config and not isinstance(config[param], float):
            raise TypeError("parameters {0} must be float".format(param))

    str_parameters = ['train_file', 'eval_file', 'test_file', 'infer_file', 'method', \
                      'load_model_name', 'loss', 'optimizer', 'init_method', 'attention_activation']
    for param in str_parameters:
        if param in config and not isinstance(config[param], str):
            raise TypeError("parameters {0} must be str".format(param))

    list_parameters = ['layer_sizes', 'activation', 'dropout']
    for param in list_parameters:
        if param in config and not isinstance(config[param], list):
            raise TypeError("parameters {0} must be list".format(param))

    if ('data_format' in config) and (not config['data_format'] in ['ffm', 'din', 'cccfnet']):
        raise TypeError("parameters data_format must be din" \
                        ",ffm, cccfnet but is {0}".format(config['data_format']))


def check_nn_config(config):
    """check neural networks config"""
    if config['model']['model_type'] in ['fm']:
        required_parameters = ['train_file', 'eval_file', 'FEATURE_COUNT', 'dim', 'loss', 'data_format', 'method']
    elif config['model']['model_type'] in ['lr']:
        required_parameters = ['train_file', 'eval_file', 'FEATURE_COUNT', 'loss', 'data_format', 'method']
    elif config['model']['model_type'] in ['din']:
        required_parameters = ['train_file', 'eval_file', 'PAIR_NUM', 'DNN_FIELD_NUM', 'FEATURE_COUNT', 'dim', \
                               'layer_sizes', 'activation', 'attention_layer_sizes', 'attention_activation', 'loss', \
                               'data_format', 'dropout', 'method']
    elif config['model']['model_type'] in ['cccfnet']:
        required_parameters = ['train_file', 'eval_file', 'dim', 'layer_sizes', 'n_user', 'n_item', 'n_user_attr',
                               'n_item_attr',
                               'activation', 'loss', 'data_format', 'dropout', 'mu', 'method']
    elif config['model']['model_type'] in ['exDeepFM']:
        required_parameters = ['train_file', 'eval_file', 'FIELD_COUNT', 'FEATURE_COUNT', 'method',
                               'dim', 'layer_sizes', 'cross_layer_sizes', 'activation', 'loss', 'data_format', 'dropout']
    elif config['model']['model_type'] in ['deepcross']:
        required_parameters = ['train_file', 'eval_file', 'FIELD_COUNT', 'FEATURE_COUNT', 'method',
                               'dim', 'layer_sizes', 'cross_layers', 'activation', 'loss', 'data_format',
                               'dropout']
    else:
        required_parameters = ['train_file', 'eval_file', 'FIELD_COUNT', 'FEATURE_COUNT', 'method',
                               'dim', 'layer_sizes', 'activation', 'loss', 'data_format', 'dropout']
    f_config = flat_config(config)
    # check required parameters
    for param in required_parameters:
        if param not in f_config:
            raise ValueError("parameters {0} must be set".format(param))
    if f_config['model_type'] == 'din':
        if f_config['data_format'] != 'din':
            raise ValueError(
                "for din model, data format must be din, but your set is {0}".format(f_config['data_format']))
    elif f_config['model_type'] == 'cccfnet':
        if f_config['data_format'] != 'cccfnet':
            raise ValueError(
                "for cccfnet model, data format must be cccfnet, but your set is {0}".format(f_config['data_format']))
    else:
        if f_config['data_format'] != 'ffm':
            raise ValueError("data format must be ffm, but your set is {0}".format(f_config['data_format']))
    check_type(f_config)


def check_config(config):
    """check networks config"""
    if config['model']['model_type'] not in ['deepFM', 'deepWide', 'dnn', 'ipnn', \
                                             'opnn', 'fm', 'lr', 'din', 'cccfnet', 'deepcross', 'exDeepFM', "cross"]:
        raise ValueError(
            "model type must be cccfnet, deepFM, deepWide, dnn, ipnn, opnn, fm, lr, din, deepcross, exDeepFM, "
            "cross, but you set is {0}".format(config['model']['model_type']))
    check_nn_config(config)


# train process load yaml
def load_yaml():
    """load config from yaml"""
    yaml_name = util.CONFIG_DIR + util.TRAIN_YAML
    print('trainging network configuration file is {0}'.format(yaml_name))
    util.check_file_exist(yaml_name)
    config = util.load_yaml_file(yaml_name)
    return config


def main():
    """main function"""

    # init
    from mx_rec.util.initialize import init
    init(use_dynamic=True,
         use_dynamic_expansion=False)

    util.check_tensorflow_version()
    util.check_and_mkdir()
    config = load_yaml()
    check_config(config)
    hparams = create_hparams(config)
    print(hparams.values())
    log = Log(hparams)
    hparams.logger = log.logger
    train.train(hparams)


main()

