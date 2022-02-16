import pandas as pd 
from joblib import dump
from rich.console import Console

from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

console = Console()

# Load the training data
df = pd.read_csv("clinc_oos-plus.csv").loc[lambda d: d['split'] == 'train']
console.log("Training data loaded.")

X = df['text'].to_list()
y = df['label']

# Make a very basic machine learning pipeline
pipe = make_pipeline(
    make_union(
        CountVectorizer(),
    ), 
    LogisticRegression()
)

pipe.fit(X, y)
console.log("ML Pipeline fitted.")

dump(pipe, 'pipe.joblib')
console.log("Joblib pickle saved.")

# Convert everything to the onnx format.
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType


initial_type = [('float_input', StringTensorType([None, 1]))]
onx = convert_sklearn(pipe, initial_types=initial_type)

with open("clinc-logreg.onnx", "wb") as f:
    f.write(onx.SerializeToString())
console.log("ONNX format saved.")
