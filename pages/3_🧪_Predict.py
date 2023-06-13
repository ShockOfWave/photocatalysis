import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from src.utils import get_project_path
from src.models.predict_model import load_model, load_sc, load_les

@st.cache_resource
def load_preprocessing_and_model():
    model = load_model()
    sc = load_sc()
    return model, sc

@st.cache_data
def load_preprocessed_data():
    preprocessed_data = pd.read_csv(os.path.join(get_project_path(), 'data', 'processed', 'data.csv'))
    return preprocessed_data

plt.style.use('ggplot')

st.set_page_config(
    page_title='Predict',
    page_icon="🧪",
)

st.markdown('# Predict')
st.sidebar.header('Predict')

model, sc = load_preprocessing_and_model()
le_dob, le_donor, le_method, le_prec = load_les()
processed_data = load_preprocessed_data()

st.write(
    """
    On this page you can predict with our model
    """
)

st.markdown('### Select parameters')
col1, col2 = st.columns(2)

with col1:
    char_1 = st.slider('Precursor', 0, (len(le_prec.classes_)-1), 0)
    st.write(f'You select {le_prec.inverse_transform([char_1])[0]}')
    char_2 = st.slider('Additive', 0, (len(le_dob.classes_)-1), 0)
    st.write(f'You select {le_dob.inverse_transform([char_2])[0]}')
    
with col2:
    char_3 = st.slider('Pt application method', 0, (len(le_method.classes_)-1), 0)
    st.write(f'You select {le_method.inverse_transform([char_3])[0]}')
    char_4 = st.slider('Electron donor', 0, (len(le_donor.classes_)-1), 0)
    st.write(f'You select {le_donor.inverse_transform([char_4])[0]}')
    
if st.button("Predict W(H2)"):
    tmp_dataframe = deepcopy(processed_data)
    tmp_dataframe[tmp_dataframe.columns[0]] = [char_1 for _ in range(len(tmp_dataframe[tmp_dataframe.columns[0]]))]
    tmp_dataframe[tmp_dataframe.columns[1]] = [char_2 for _ in range(len(tmp_dataframe[tmp_dataframe.columns[1]]))]
    tmp_dataframe[tmp_dataframe.columns[2]] = [char_3 for _ in range(len(tmp_dataframe[tmp_dataframe.columns[2]]))]
    tmp_dataframe[tmp_dataframe.columns[8]] = [char_4 for _ in range(len(tmp_dataframe[tmp_dataframe.columns[8]]))]
    tmp_dataframe.drop(tmp_dataframe.columns[-1], axis=1, inplace=True)
    tmp_data = sc.transform(tmp_dataframe)
    result = model.predict(tmp_data)
    max_value = max(result)
    index_for_df = np.where(result == max_value)[0][0]
    best_result = tmp_dataframe.loc[index_for_df]
    # best_result = best_result.to_frame()
    best_result['W(H2)^b [μmol/min]'] = max_value
    best_result[best_result.index[0]] = le_prec.inverse_transform([int(best_result[best_result.index[0]])])
    best_result[best_result.index[1]] = le_dob.inverse_transform([int(best_result[best_result.index[1]])])
    best_result[best_result.index[2]] = le_method.inverse_transform([int(best_result[best_result.index[2]])])
    best_result[best_result.index[8]] = le_donor.inverse_transform([int(best_result[best_result.index[8]])])
    
    st.markdown(f'''
                ### predicted W(H2)^b [μmol/min] is {max_value}
                #### There is best parameters
                ''')
    
    st.dataframe(best_result)
    