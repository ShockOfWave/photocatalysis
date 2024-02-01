import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.utils import check_password


if not check_password():
    st.stop()

plt.style.use('ggplot')

st.set_page_config(
    page_title='Dataset',
    page_icon="📊",
)

st.markdown('# Dataset')

st.write(
    """
    On this page we show you our dataset and our preprocessing
    """
)

st.markdown(
    """
    ### Raw data
    There is our raw data
    """
)


@st.cache_data
def load_data():
    raw_data = pd.read_csv('data/raw/data.csv', delimiter=';')
    processed_data = pd.read_csv('data/processed/data.csv')
    return raw_data, processed_data


raw_data, processed_data = load_data()

st.dataframe(raw_data)

st.markdown(
    """
    ### Processed data
    We encode all string values to make it readable for machine learning
    """
)

st.dataframe(processed_data)

st.markdown(
    """
    ### Correlation matrix
    Correlation matrix with processed data
    """
)

fig, ax = plt.subplots(figsize=(12, 9))
matrix = processed_data.corr().round(2)
mask = np.triu(np.ones_like(processed_data.corr().round(2)))
sns.heatmap(matrix, annot=True, ax=ax, cmap='Blues', mask=mask)
ax.set_title('Correlation matrix')
ax.tick_params(axis='x', rotation=45)
ax.set_xticklabels(processed_data.columns, ha='right')
fig.tight_layout()
st.pyplot(fig)

st.markdown(
    """
    ### Histograms
    Histograms for each column in processed data
    """
)

fig, ax = plt.subplots(figsize=(20, 18))
processed_data.hist(ax=ax)
ax.set_title('Histogram XRD1 2theta')
ax.set_ylabel('Num features')
ax.set_xlabel('Feature value')
st.pyplot(fig)