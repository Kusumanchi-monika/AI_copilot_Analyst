import streamlit as st
import os
from dotenv import load_dotenv, set_key

load_dotenv()

st.title("Environment Variables Settings")

# Function to update .env file
def update_env_file(key, value):
    set_key(".env", key, value)
    os.environ[key] = value

# Input fields for environment variables
openai_api_key = st.text_input("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY") or "")
mongodb_uri = st.text_input("uri", os.getenv("uri") or "")
credentials_path = st.text_input("credentials_path", os.getenv("credentials_path") or "")
sheet_id = st.text_input("sheet_id", os.getenv("sheet_id") or "")

if st.button("Save Settings"):
    update_env_file("OPENAI_API_KEY", openai_api_key)
    update_env_file("uri", mongodb_uri)
    update_env_file("credentials_path", credentials_path)
    update_env_file("sheet_id", sheet_id)
    st.success("Settings saved!")