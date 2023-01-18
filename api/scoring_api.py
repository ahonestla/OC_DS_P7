import json
import pickle
import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import sklearn

app = FastAPI()

# Unpick classifier
clf = pickle.load(open('randomforest_v1.pckl', 'rb'))

# Get parameters
params = clf.get_params(deep=True)


@app.get('/')
def hello():
    return "Hello There!"


@app.get("/test")
def test():
    return json.dumps(params)

if __name__ == "__main__":
    uvicorn.run("scoring_api:app", host="0.0.0.0", port=8000)