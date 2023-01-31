import json
import pickle
import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import sklearn

app = FastAPI()

# Unpick classifier
# clf = pickle.load(open('randomforest_v1.pckl', 'rb'))
clf = pickle.load(open('./models/randomforest_v1.pckl', 'rb'))

# Get parameters
params = clf.get_params(deep=True)

# Unpick test data
test_df = pickle.load(open('./data/processed/test_feature_engineering.pckl', 'rb'))
print(test_df.head())

@app.get('/')
def main():
    return "Hello There! This is the front page of the scoring API."


@app.get("/test")
def test():
    return json.dumps(params)


@app.get("/str")
def test():
    return {'str': 'coucou'}


# Return the test ids
@app.get("/ids")
def ids():
    return {'ids': test_df.head().index.to_list()}

if __name__ == "__main__":
    uvicorn.run("scoring_api:app", reload=True, host="0.0.0.0", port=8000)
    # uvicorn.run("scoring_api:app", host="0.0.0.0", port=8000)