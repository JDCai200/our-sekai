"""Same named-tensor interface for desktop parity and Android Java ONNX."""
from pathlib import Path
import base64
import json
import numpy as np


class Runtime:
    def __init__(self,models,android=False):
        self.models=Path(models); self.sessions={}; self.android=android
        if android:
            from java import jclass
            self.bridge=jclass('org.oursekai.companion.OnnxBridge')()

    def run(self,name,inputs):
        path=self.models/(name+'.onnx')
        inputs={key:np.ascontiguousarray(value) for key,value in inputs.items()}
        if self.android:
            payload=[dict(name=key,shape=list(value.shape),int64=value.dtype==np.int64,
                data=base64.b64encode(value.tobytes()).decode('ascii')) for key,value in inputs.items()]
            outputs=json.loads(str(self.bridge.runJson(str(path),json.dumps(payload))))
            return [np.frombuffer(base64.b64decode(item['data']),dtype='<f4').reshape(item['shape']).copy() for item in outputs]
        if name not in self.sessions:
            import onnxruntime as ort
            options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
            self.sessions[name]=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
        return self.sessions[name].run(None,inputs)

    def release(self,name):
        if self.android:self.bridge.release(str(self.models/(name+'.onnx')))
        else:self.sessions.pop(name,None)

    def close(self):
        if self.android:self.bridge.close()
        self.sessions.clear()
