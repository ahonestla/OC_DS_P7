import json
import pickle
import dill
import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import sklearn

app = FastAPI()

test_df = pd.read_csv("./data/processed/test_feature_engineering.csv", index_col=[0])
print(test_df.head())

# Unpick classifier
# xgbc = pickle.load(open('./models/xgboost_tuned_v1.pckl', 'rb'))
xgbc = pickle.load(open('./models/xgboost_default_v2.pckl', 'rb'))
params = xgbc.get_params(deep=True)

# Unpick explainer
with open('./models/xgboost_explainer.pckl', 'rb') as f:
    explainer = dill.load(f)

@app.get('/')
def main():
    """ API main page """
    return "Hello There! This is the front page of the scoring API."


@app.get("/test")
def test():
    """ API test page """
    return json.dumps(params)


@app.get("/ids")
def ids():
    """ Return the customers ids """
    return {'ids': test_df.head().index.to_list()}


@app.get("/explain/id={cust_id}")
def explain(cust_id: int):
    """ Return the customer id explanation as html """
    print("Prediction : ", xgbc.predict(test_df.to_numpy())[cust_id])
    explanation = explainer.explain_instance(test_df.to_numpy()[cust_id], xgbc.predict_proba).as_html()
    return {'explanation': explanation}


if __name__ == "__main__":
    uvicorn.run("scoring_api:app", reload=True, host="0.0.0.0", port=8000)