import streamlit as st
import os
import json
from dotenv import load_dotenv, set_key

load_dotenv()

st.title("Environment Variables Settings")

# Function to update .env file
def update_env_file(key, value):
    set_key(".env", key, value)
    os.environ[key] = value

# Input fields for environment variables (masked)
openai_api_key = st.text_input("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY") or "", type="password")
mongodb_uri = st.text_input("uri", os.getenv("uri") or "", type="password")
sheet_id = st.text_input("sheet_id", os.getenv("sheet_id") or "", type="password")
groq_api_key = st.text_input("GROQ_API_KEY", os.getenv("sheet_id") or "", type="password")

        
if st.button("Save Settings"):
    update_env_file("OPENAI_API_KEY", openai_api_key)
    update_env_file("uri", mongodb_uri)
    update_env_file("sheet_id", sheet_id)
    update_env_file("GROQ_API_KEY", sheet_id)
    

    
    st.success("Settings saved!")
