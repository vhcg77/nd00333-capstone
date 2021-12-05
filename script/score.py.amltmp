import json
import pandas as pd
import numpy as np
import os
import joblib, pickle
from azureml.core import Model


def init():
    global model
    model_path = Model.get_model_path('hyperdrive_model')
    model = joblib.load(model_path)

def run(data):
    try:
        input_data = json.loads(data)
        json_data = input_data['data']
        dataframe = pd.DataFrame(json_data)
        result = model.predict(dataframe)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error