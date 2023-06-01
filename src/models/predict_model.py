import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

def load_model():
    model = CatBoostRegressor(task_type='GPU', devices='1')
    model.load_model('../../models/serialized/model')
    return model

def predict_model(model, data):
    return model.predict(data)