import os
import streamlit as st
import pandas as pd
import numpy as np
from copy import deepcopy
from src.utils import get_project_path
from src.models.predict_model import load_model, load_process, load_les
from src.utils import check_password


if not check_password():
    st.stop()


@st.cache_resource
def load_preprocessing_and_model():
    model = load_model()
    sc = load_process()
    return model, sc


@st.cache_data
def load_preprocessed_data():
    preprocessed_data = pd.read_csv(os.path.join(get_project_path(), 'data', 'processed', 'data.csv'))
    return preprocessed_data


st.markdown('''
            # Model predictions
            ##### On this page you can select your components and predict best output for system
            ''')

model, sc = load_preprocessing_and_model()
le_dob, le_donor, le_method, le_prec = load_les()
processed_data = load_preprocessed_data()

st.markdown('##### Select your parameters')

col1, col2 = st.columns(2)


with col1:
    char_1 = st.selectbox(
        "Select precursor",
        le_prec.classes_
    )

    char_2 = st.selectbox(
        'Select additive',
        le_dob.classes_
    )

with col2:
    char_3 = st.selectbox(
        "Select Pt application method",
        le_method.classes_
    )

    char_4 = st.selectbox(
        'Select electron donor',
        le_donor.classes_
    )

char_1 = le_prec.transform([char_1])[0]
char_2 = le_dob.transform([char_2])[0]
char_3 = le_method.transform([char_3])[0]
char_4 = le_donor.transform([char_4])[0]

if st.button("Predict W(H2)"):
    tmp_dataframe = deepcopy(processed_data)
    tmp_dataframe[tmp_dataframe.columns[0]] = [char_1 for _ in range(len(tmp_dataframe[tmp_dataframe.columns[0]]))]
    tmp_dataframe[tmp_dataframe.columns[1]] = [char_2 for _ in range(len(tmp_dataframe[tmp_dataframe.columns[1]]))]
    tmp_dataframe[tmp_dataframe.columns[2]] = [char_3 for _ in range(len(tmp_dataframe[tmp_dataframe.columns[2]]))]
    tmp_dataframe[tmp_dataframe.columns[8]] = [char_4 for _ in range(len(tmp_dataframe[tmp_dataframe.columns[8]]))]
    tmp_dataframe.drop(tmp_dataframe.columns[-1], axis=1, inplace=True)
    tmp_data = sc.transform(tmp_dataframe)
    result = np.round(model.predict(tmp_data), 6)
    max_value = max(result)
    index_for_df = np.where(result == max_value)[0][0]
    best_result = tmp_dataframe.loc[index_for_df]
    best_result['W(H2)^b [μmol/min]'] = max_value
    best_result[best_result.index[0]] = le_prec.inverse_transform([int(best_result[best_result.index[0]])])
    best_result[best_result.index[1]] = le_dob.inverse_transform([int(best_result[best_result.index[1]])])
    best_result[best_result.index[2]] = le_method.inverse_transform([int(best_result[best_result.index[2]])])
    best_result[best_result.index[8]] = le_donor.inverse_transform([int(best_result[best_result.index[8]])])
    
    st.markdown(f'''
                ### Predicted W(H2)^b [μmol/min] is {max_value}
                #### There is best parameters
                ''')
    
    st.dataframe(best_result, height=((len(best_result) + 1) * 35 + 3))
