import time 
from joblib import load

import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession("clinc-logreg.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pipe = load('pipe.joblib')

text = "this is an example sentence"
n = 1000

t0 = time.time()
for i in range(n):
    pipe.predict_proba([text])
t1 = time.time()
for i in range(n):
    _, probas = sess.run(None, {input_name: np.array([[text]])})
t2 = time.time()

print(f"SKLEARN: {round(t1 - t0, 3)} s")
print(f"   ONNX: {round(t2 - t1, 3)} s")
