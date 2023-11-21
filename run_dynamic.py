# !/usr/bin/env python3
import os
from time import perf_counter
import json

import onnx
import onnxruntime
import numpy as np

from openvino.runtime import Core, Model, Tensor, PartialShape, Type, ProfilingInfo, properties
from openvino.runtime import opset8 as opset
from openvino.runtime.op import Constant, Parameter, tensor_iterator
from openvino.runtime.passes import Manager
from openvino.runtime.utils.types import get_dtype
import openvino as ov

print_config = True
enable_onnx_profiling = True
enable_openvino_profiling = False
enable_save_data = False

MAX_SEQ = 64
#MAX_SEQ = 12
#MAX_SEQ = 13
bos_token_id = 0
eos_token_id = 2

def onnx_infer(sess, input_ids, position_ids, attention_mask, past_key_values):
    """
    """
    input_dict = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values
    }
    output_name_list = []
    for node in sess.get_outputs():
        output_name_list.append(node.name)
    res = sess.run(output_name_list, input_dict)
    return res

def get_token(next_tokens_scores):
    """
    简单实现，用以验证模型。
    只保证输入输出与原始模型一致。
    """
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1)
    probs = softmax(next_tokens_scores)
    probs = np.log(probs)
    next_tokens = np.argmax(probs, axis=-1).reshape([-1, 1])
    return next_tokens

def save_data(engine, q_mode, cur_len, outputs):
    folder_name = f"./result/{engine}_{q_mode}_{cur_len}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    np.save(f"{folder_name}/logits", outputs[0][:, -1, :])
    np.save(f"{folder_name}/past_key_values", outputs[1])

def main(engine = "onnx", q_mode = False):
    print(f"Engine: {engine}")

    # text = "Hello, it is a great day! nice to"
    input_ids = np.array([[1, 16644, 31844, 357, 322, 260, 1014, 1124, 31905, 4504, 289]], dtype=np.int64)
    position_ids = np.array([[7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87]], dtype=np.int64)
    cur_len = input_ids.shape[-1] # 11
    attention_mask = np.ones([1, 1, cur_len, cur_len], dtype=np.float32) * np.finfo(np.float32).min
    attention_mask = np.triu(attention_mask, 1) # 返回输入矩阵 input 的上三角部分，其余部分被设为 0。
    past_key_values = np.zeros([31, 2, 0, 24, 128], "float32") # 实际上是个空矩阵

    if engine == "onnx":
        sess_options = None
        if enable_onnx_profiling:
            sess_options = onnxruntime.SessionOptions()
            sess_options.enable_profiling = True
            sess_options.profile_file_prefix = "/home/openvino-ci-58/meiyang/work/ernie3.5se/dynamic/onnx_profiling/onnx_profiling"
        model_file = "/mnt/llm_irs/paddle/dynamic_model_onnx/model.onnx"
        sess = onnxruntime.InferenceSession(model_file, sess_options)
    elif engine == "ov":
        if q_mode:
            model_file = f"/mnt/llm_irs/paddle/{q_mode}/model.xml"
        else:
            model_file = "/mnt/llm_irs/paddle/ir/model.xml"
        core = ov.Core()
        model = core.read_model(model_file)
        latency = {'PERFORMANCE_HINT': 'LATENCY',
                   'SCHEDULING_CORE_TYPE': 'PCORE_ONLY',
                   'ENABLE_HYPER_THREADING': 'YES',
                  }
        if enable_openvino_profiling:
            latency['PERF_COUNT'] = 'YES'
        compiled_model = core.compile_model(model, 'CPU', latency)
        if print_config:
            keys = compiled_model.get_property(properties.supported_properties())
            print("Model:")
            for k in keys:
                skip_keys = ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', properties.supported_properties())
                if k not in skip_keys:
                    value = compiled_model.get_property(k)
                    print(f'    {k}: {value}')
                    #if k == properties.device.properties():
                    #for device_key in value.keys():
                    #    print(f'  {device_key}:')
                    #    for k2, value2 in value.get(device_key).items():
                    #        if k2 not in skip_keys:
                    #            print(f'    {k2}: {value2}')

        ireq = compiled_model.create_infer_request()

    print("load success!")
    # 超过最大生成长度 或者 遇到结束符
    infer_time = []
    nodes_list = []
    while cur_len < MAX_SEQ:
        start = perf_counter()
        if engine == "onnx":
            outputs = onnx_infer(
                sess,
                input_ids,
                position_ids,
                attention_mask,
                past_key_values
            )
        elif engine == "ov":
            inputs = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values
            }
            outputs = ireq.infer(inputs)
            if enable_openvino_profiling:
                pi_list = []
                for profile_info in ireq.get_profiling_info():
                    status = "OPTIMIZED_OUT"
                    if profile_info.status == ProfilingInfo.Status.EXECUTED:
                        status = "EXECUTED"
                    elif profile_info.status == ProfilingInfo.Status.NOT_RUN:
                        status = "NOT_RUN"
                    pi = {"name": profile_info.node_name,
                          "cpu_time": profile_info.cpu_time.microseconds/1000.0,
                          "real_time": profile_info.real_time.microseconds/1000.0,
                          "node_type": profile_info.node_type,
                          "exec_type": profile_info.exec_type,
                          "status": status}
                    pi_list.append(pi)
                nodes_list.append({"nodes": pi_list})
        elapsed = perf_counter() - start
        infer_time.append(elapsed)

        cur_len += 1
        logits = outputs[0][:, -1, :]
        past_key_values = outputs[1]
        next_token = get_token(logits) # (1, 1)
        print("next_token: ", next_token)
        # 结束符
        if next_token == eos_token_id:
            print("eos")
            break

        input_ids = next_token
        position_ids = np.array([[cur_len - 1]], dtype=np.int64)
        attention_mask = np.zeros([1, 1, 1, cur_len], dtype=np.float32)

        if enable_save_data:
            save_data(engine, q_mode, cur_len, outputs)
    if engine == "ov" and enable_openvino_profiling:
        dp = {"detailed_performance": nodes_list, "report_type": "detailed"}
        with open("benchmark_detailed_counters_report.json", "w") as f:
            json.dump(dp, f)
    #print(infer_time)
    infer_time.pop(0)
    if len(infer_time):
        avg_time = sum(infer_time) / len(infer_time)
        print(avg_time)

if __name__ == '__main__':
    #main("onnx")
    main("ov")
    #main("ov", "INT8")
    #main("ov", "INT4_SYM")
