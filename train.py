import pandas as pd 

from joblib import dump

from sklearn.pipeline import make_pipeline, make_union
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

# Load the training data
df = pd.read_csv("clinc_oos-plus.csv").loc[lambda d: d['split'] == 'train']

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

# Convert everything to the onnx format.
initial_type = [('float_input', StringTensorType([None, 1]))]
onx = convert_sklearn(pipe, initial_types=initial_type)

with open("clinc-logreg.onnx", "wb") as f:
    f.write(onx.SerializeToString())

dump(pipe, 'pipe.joblib')
