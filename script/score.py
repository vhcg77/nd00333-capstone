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
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error