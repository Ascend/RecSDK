import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc
import numpy as np
import os
import time
 
class PredictModelGrpc(object):
    def __init__(self, model_name, inputs,input_types, output_name, socket='xxx.xxx.xxx.xxx:8500'):# xxx.xxx.xxx.xxx为服务端IP地址
        self.socket = socket
        self.model_name = model_name
        self.inputs = inputs
        self.input_types = input_types
        self.output_name = output_name
        self.request, self.stub = self.__get_request()
    def __get_request(self):
        channel = grpc.insecure_channel(self.socket, options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                                                              ('grpc.max_receive_message_length',
                                                               1024 * 1024 * 1024)])  # 可设置大小
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
 
        request.model_spec.name = self.model_name
        request.model_spec.signature_name = "serving_default"
 
        return request, stub
    def inference(self):
 
        t0 = time.time()
        for name in self.inputs:
            self.request.inputs[name].CopyFrom(tf.make_tensor_proto(self.inputs[name], dtype=self.input_types[name]))# 发送请求
        t1 = time.time()
        print("request.inputs={:.3f} ms".format((t1 - t0) * 1000))
        
        for i in range(100):
            result = self.stub.Predict.future(self.request, 1000.0)
            result.result()
            t2 = time.time()
            print("request serving time cost: {:.3f} ms".format((t2 - t1) * 1000))
            t1 = t2
 
 
        res = []
        #print(result.result().outputs['task_1'])
        #print(result.result().outputs['task_2'])
        #res.append(tf.make_ndarray(result.result().outputs[self.output_name])[0]) # 获取结果
 
        t3 = time.time()
        return res
 
def gen_inputs():
    from input_config import config
    inputs = {}
    input_types = {}
    for name in config:
        input_types[name] = config[name]['dtype']
        if config[name]['dtype'] == tf.int32:
            inputs[name] = np.random.randint(0,100,size=config[name]['shape'])
        elif config[name]['dtype'] == tf.float32:
            inputs[name] = np.random.randint(0,2,size=config[name]['shape'])*1.0
    return inputs, input_types
if __name__ == '__main__':
    inputs,input_types = gen_inputs()
    model = PredictModelGrpc(model_name='saved_model', inputs = inputs, input_types = input_types, output_name='MobilenetV2/Logits/output:0',socket='127.0.0.1:9999') 
 
    res = model.inference()
