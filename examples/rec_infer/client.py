#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024. Huawei Technologies Co.,Ltd. All rights reserved.
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

import os
import time

import grpc
import numpy as np

import tensorflow as tf
from input_config import config
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc


class PredictModelGrpc():
    def __init__(
        self,
        model_name,
        inputs,
        input_types,
        output_name,
        socket="xxx.xxx.xxx.xxx:8500",
    ):
        self.socket = socket
        self.model_name = model_name
        self.inputs = inputs
        self.input_types = input_types
        self.output_name = output_name
        self.request, self.stub = self.__get_request()

    def inference(self):
        for name in self.inputs:
            self.request.inputs[name].CopyFrom(
                tf.make_tensor_proto(self.inputs[name], dtype=self.input_types[name])
            )

        for _ in range(100):
            result = self.stub.Predict.future(self.request, 1000.0)
            result.result()

    def __get_request(self):
        channel = grpc.insecure_channel(
            self.socket,
            options=[
                ("grpc.max_send_message_length", 1024 * 1024 * 1024),
                ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ],
        )
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"

        return request, stub


FIELD_TYPE = "dtype"
FIELD_SHAPE = "shape"


def gen_inputs():
    inputs = {}
    input_types = {}
    for name in config:
        input_types[name] = config[name][FIELD_TYPE]
        if config[name][FIELD_TYPE] == tf.int32:
            inputs[name] = np.random.randint(0, 100, size=config[name][FIELD_SHAPE])
        elif config[name][FIELD_TYPE] == tf.float32:
            inputs[name] = np.random.randint(0, 2, size=config[name][FIELD_SHAPE]) * 1.0
    return inputs, input_types


if __name__ == "__main__":
    input_datas, types = gen_inputs()
    model = PredictModelGrpc(
        model_name="saved_model",
        inputs=input_datas,
        input_types=types,
        output_name="",
        socket="127.0.0.1:9999",
    )

    model.inference()
