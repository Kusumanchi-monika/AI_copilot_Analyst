import streamlit as st
from pymongo import MongoClient
import json
from dotenv import load_dotenv
import os

load_dotenv()
uri = os.getenv("uri")


@st.cache_resource
def get_mongo_client():
    return MongoClient(uri)

client = get_mongo_client()


st.title("Upload JSON Data to MongoDB")

db_name = st.text_input("Database Name:")
collection_name = st.text_input("Collection Name:")
json_file = st.file_uploader("Upload JSON file", type=["json"])

if st.button("Upload Data"):
    if not db_name or not collection_name or not json_file:
        st.error("Please provide all the required information.")
    else:
        try:
            data = json.load(json_file)

            db = client[db_name]
            collection = db[collection_name]

            if isinstance(data, list):
                collection.insert_many(data)
            else:
                collection.insert_one(data)

            st.success("Data uploaded successfully!")
        except Exception as e:
            st.error(f"An error occurred: {e}")