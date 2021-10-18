from pydantic import BaseModel
from fastapi import FastAPI
from joblib import load

app = FastAPI()

class Query(BaseModel):
    text: str


import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession("clinc-logreg.onnx")
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pipe = load('pipe.joblib')


@app.get("/")
def root():
    return {"health": "alive"}


@app.post("/predict")
def predict(query: Query):
    """Classifies text"""
    _, probas = sess.run(None, {input_name: np.array([[query.text]])})
    return probas[0]


@app.post("/sk_predict")
def sk_predict(query: Query):
    """Classifies text"""
    probas = pipe.predict_proba([query.text])
    return {int(k):float(p) for k, p in zip(pipe.classes_, probas[0])}
