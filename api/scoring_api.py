import json
from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Unpick classifier
clf = pickle.load(open('../models/randomforest_v1.pckl', 'rb'))

# Get parameters
params = clf.get_params(deep=True)


@app.route("/test")
def test():
    # print(data)
    return json.dumps(params)

if __name__ == '__main__':
    app.run(debug=True)