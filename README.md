# test step
## 1. install openvino python

## 2. setup mo
```
cd openvino/tools/mo/
pip3 install -r requirements_onnx.txt requirements.txt
python ./setup.py install
```

## 3. convert ernie3.5se onnx model to ir model
```
python3 -- <venv>/bin/mo --framework=onnx --output_dir=./ir --input_model=/home/iot/tmp/ernie3.5se/dynamic_model_onnx/model.onnx --compress_to_fp16=False
```

## 4. benchmark
    Replace variable 'model_file' in run_dynamic.py with converted ir model and run.
```
python3 run_dynamic.py
```

# run paddle model
If want to run paddle model directly with openvino, just replace paddle model as below.
```
--- a/run_dynamic.py
+++ b/run_dynamic.py
@@ -85,7 +85,7 @@ def main(engine = "onnx", q_mode = False):
         else:
             model_file = "/mnt/llm_irs/paddle/ir/model.xml"
         core = ov.Core()
-        model = core.read_model(model_file)
+        model = core.read_model("abc.pdmodel")
         latency = {'PERFORMANCE_HINT': 'LATENCY',
                    'SCHEDULING_CORE_TYPE': 'PCORE_ONLY',
                    'ENABLE_HYPER_THREADING': 'YES',
```
