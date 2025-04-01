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

# Input fields for environment variables
openai_api_key = st.text_input("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY") or "")
mongodb_uri = st.text_input("uri", os.getenv("uri") or "")
sheet_id = st.text_input("sheet_id", os.getenv("sheet_id") or "")

# Text area for service account JSON
direct_json_input = st.text_area("Service Account JSON", "")

if st.button("Save Settings"):
    update_env_file("OPENAI_API_KEY", openai_api_key)
    update_env_file("uri", mongodb_uri)
    update_env_file("sheet_id", sheet_id)
    
    # Save the provided service account JSON
    if direct_json_input:
        try:
            credentials_json = json.loads(direct_json_input)
            credentials_path = os.path.join(os.getcwd(), "credentials.json")  # Save to main directory
            with open(credentials_path, "w") as f:
                json.dump(credentials_json, f, indent=4)
            st.success("Credentials file saved successfully!")
        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check your input.")
    
    st.success("Settings saved!")