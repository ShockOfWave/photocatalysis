import streamlit as st
from src.utils import show_code

st.set_page_config(
    page_title='Photocatalysis',
    page_icon='🌼',
)

st.write('# Welcome to application 🌼')

st.sidebar.header('Welcome page')

st.markdown(
    """
    This application was created for testing model for our Data Science project 🔥
    **Select any page for see our model**
    
    ### About us
    - We are from [ITMO University](https://itmo.ru) and [Infochemistry Scientific Center (ISC)](https://infochemistry.ru)
    """
)