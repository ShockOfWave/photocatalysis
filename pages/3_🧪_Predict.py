import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models.predict_model import load_model
from src.models.predict_model import load_sc

@st.cache
def load_preprocessing_and_model():
    model = load_model()
    sc = load_sc()
    return model, sc

plt.style.use('ggplot')

st.set_page_config(
    page_title='Predict',
    page_icon="🧪"
)

st.markdown('# Predict')
st.sidebar.header('Predict')

model, sc = load_preprocessing_and_model()

st.write(
    """
    On this page you can predict with our model
    """
)

st.header('Parameters')
col1, col2 = st.columns(2)

with col1:
    st.text('First char')
    char_1 = st.slider('Sepal')