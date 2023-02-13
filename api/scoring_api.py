import json
import pickle
import dill
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

app = FastAPI()
N_CUSTOMERS = 100
MAIN_COLUMNS = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 
                'NAME_FAMILY_STATUS_Married', 'NAME_INCOME_TYPE_Working',
                'AMT_INCOME_TOTAL', 'PAYMENT_RATE']

# Get test dataframe
test_df = pd.read_csv("./data/processed/test_feature_engineering.csv", index_col=[0])
test_columns = test_df.columns

# Unpick XGB classifier
xgbc = pickle.load(open('./models/xgboost_classifier.pckl', 'rb'))

# Unpick SHAP explainer
explainer = pickle.load(open('./models/xgboost_shap_explainer.pckl', 'rb'))


def prepare_data(data, n_customers):
    """ Prepare the data and compute the shap values """
    data = data.iloc[0: n_customers]

    # Fill values with imputer
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(data)
    values = imp_mean.transform(data)

    # Compute shap values
    shap_values = explainer(values)

    # Create new df
    df = pd.DataFrame(values, columns=data.columns)

    return df, shap_values


# Prepare the reduced data and shap values
prep_df, shap_values = prepare_data(test_df, N_CUSTOMERS)


@app.get('/')
def main():
    """ API main page """
    return "Hello There! This is the front page of the scoring API."


@app.get("/ids")
def ids():
    """ Return the customers ids """
    return {'ids': test_df.head(N_CUSTOMERS).index.to_list()}


@app.get("/columns/id={cust_id}")
def columns(cust_id: int):
    """ Return the customer main columns values """
    if cust_id not in range(0, N_CUSTOMERS):
        raise HTTPException(status_code=404, detail="Customer id not found")
    cust_main_df = prep_df.iloc[cust_id][MAIN_COLUMNS]
    return cust_main_df.to_json()

@app.get("/predict/id={cust_id}")
def predict(cust_id: int):
    """ Return the customer predictions of repay """
    if cust_id not in range(0, N_CUSTOMERS):
        raise HTTPException(status_code=404, detail="Customer id not found")
    pred_default = xgbc.predict(prep_df.values)[cust_id]
    pred_custom = (xgbc.predict_proba(prep_df.values)[cust_id, 1] >= 0.7).astype(int)
    return {'default': pred_default.tolist(), 'custom': pred_custom.tolist()}


@app.get("/shap")
def explain_all():
    """ Return all shap values """
    return {'values': shap_values.values.tolist(),
            'base_values': shap_values.base_values.tolist(),
            'features': explainer.feature_names}


@app.get("/shap/id={cust_id}")
def explain(cust_id: int):
    """ Return the customer shap values """
    if cust_id not in range(0, N_CUSTOMERS):
        raise HTTPException(status_code=404, detail="Customer id not found")
    return {'values': shap_values[cust_id].values.tolist(),
            'base_values': float(shap_values[cust_id].base_values),
            'features': explainer.feature_names}


@app.get("/importances/model")
def importances():
    """ Return the model 15 top feature importances """
    imp_df = pd.DataFrame(data=xgbc.feature_importances_, index=test_columns, columns=['importances'])
    imp_df = imp_df.sort_values(by='importances', ascending=False).head(15)
    return imp_df.to_json()


if __name__ == "__main__":
    uvicorn.run("scoring_api:app", reload=True, host="0.0.0.0", port=8000)