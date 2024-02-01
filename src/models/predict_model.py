import os
import pickle
from src.utils import get_project_path


def load_model():
    model = pickle.load(open(os.path.join(get_project_path(), 'serialized_models', 'model.pkl'), 'rb'))
    return model


def predict_model(model, data):
    return model.predict(data)


def load_process():
    with open(os.path.join(get_project_path(), 'serialized_models', 'Normalizer.pkl'), 'rb') as f:
        sc = pickle.load(f)
        f.close()
    
    return sc


def load_les():
    with open(os.path.join(get_project_path(), 'serialized_models', 'le_dob.pkl'), 'rb') as f:
        le_dob = pickle.load(f)
        f.close()
    
    with open(os.path.join(get_project_path(), 'serialized_models', 'le_donor.pkl'), 'rb') as f:
        le_donor = pickle.load(f)
        f.close()
        
    with open(os.path.join(get_project_path(), 'serialized_models', 'le_method.pkl'), 'rb') as f:
        le_method = pickle.load(f)
        f.close()
        
    with open(os.path.join(get_project_path(), 'serialized_models', 'le_prec.pkl'), 'rb') as f:
        le_prec = pickle.load(f)
        f.close()  
    
    return le_dob, le_donor, le_method, le_prec
