import os
import hmac
from pathlib import Path
import streamlit as st


def get_project_path():
    """
    Reads path to the root of project
    Returns:
        path(str): path to root of project
    """
    return Path(__file__).parent.parent


PATH_TO_DATA = os.path.join(get_project_path(), 'data')
PATH_TO_MODELS = os.path.join(get_project_path(), 'models')
PATH_TO_REPORTS = os.path.join(get_project_path(), 'reports')
PATH_TO_EVALUATIONS = os.path.join(get_project_path(), 'evaluations')


def check_password():
    """Returns `True` if the user had a correct password."""
    if os.path.exists(os.path.join(get_project_path(), '.streamlit', 'secrets.toml')):
        def login_form():
            """Form with widget to collect user information"""
            with st.form("Credentials"):
                st.text_input("Username", key="username")
                st.text_input("Password", type='password', key='password')
                st.form_submit_button("Log in", on_click=password_entered)

        def password_entered():
            """Checks whether a password entered by the user is correct."""
            if st.session_state["username"] in st.secrets[
                "passwords"
            ] and hmac.compare_digest(
                st.session_state["password"],
                st.secrets.passwords[st.session_state["username"]],
            ):
                st.session_state["password_correct"] = True
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False

        if st.session_state.get("password_correct", False):
            return True

        login_form()

        if "password_correct" in st.session_state:
            st.error("User not known or password incorrect")
        return False

    else:
        pass
