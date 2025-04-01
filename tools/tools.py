from agno.tools import Toolkit 
from agno.agent import Agent
from agno.tools.thinking import ThinkingTools
import gspread
import json
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
from typing import List, Dict, Optional
from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
import os
import streamlit as st
import requests
 
sheet_id =  os.getenv("sheet_id")
uri = os.getenv("uri")

# ✅ Initialize MongoDB Utility
class MongoDBUtility(Toolkit):
    def __init__(self, uri=uri, db_name="Demo"):
        """Initialize MongoDB connection."""
        super().__init__(name="mongo_db_toolkit")
        self.uri = os.getenv("uri")
        print(f"MongoDBUtility __init__: URI from env: {self.uri}") # Debug: Show URI
        try:
            print("MongoDBUtility __init__: Trying to connect to MongoDB...")
            self.client = MongoClient(self.uri)
            print("MongoDBUtility __init__: MongoClient created.")

            self.client.admin.command('ping')  # Check connection
            print("MongoDBUtility __init__: Ping successful. MongoDB Connection Successful!")

            self.db_name = self.uri.split("/")[-1].split("?")[0]
            print(f"MongoDBUtility __init__: Database name extracted: {self.db_name}")

            self.db = self.client[db_name]
            print(f"MongoDBUtility __init__: Database object created: {self.db}")

        except Exception as e:
            print(f"MongoDBUtility __init__: ERROR Connecting to MongoDB: {e}")
            self.client = None
            self.db = None
            self.db_name = None
            st.error(f"MongoDB Connection Failed: {e}") #Showed in UI
        self.uri = os.getenv("uri")  # Access URI from environment variable
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        
        self.register(self.query_mongodb)
        self.register(self.list_collections)
        self.register(self.get_collection_schema)
        self.register(self.count_documents)
        self.register(self.find_documents)
        self.register(self.aggregate_documents)
        self.register(self.date_tool)

    def query_mongodb(self, collection, filter_query=None, projection=None, limit=10):
        """Executes a MongoDB find() query."""
        if not isinstance(filter_query, dict):
            raise TypeError("Query must be a dictionary")

        col = self.db[collection]
        cursor = col.find(filter_query or {}, projection).limit(limit)
        return str(list(cursor))

    def list_collections(self):
        """Returns the list of collections in the database."""
        collections= self.db.list_collection_names()
        return  str(collections)

    def get_collection_schema(self, collection):
        """Returns schema-like information for a collection based on sampled documents."""
        col = self.db[collection]
        sample = col.find_one()
        if sample:
            return str({field: type(value).__name__ for field, value in sample.items()})

        return "No documents found in the collection."

    def count_documents(self, collection: str, filter_query: Optional[Dict] = None) -> int:
        """Counts documents in a collection based on a filter query."""
        if not isinstance(filter_query, dict):
            raise TypeError("Filter query must be a dictionary.")

        col = self.db[collection]
        return str(col.count_documents(filter_query or {}))

    def find_documents(self, collection: str, filter_query: Optional[Dict] = None, projection: Optional[Dict] = None) -> List[Dict]:
        """Finds documents in a MongoDB collection."""
        if not isinstance(filter_query, dict):
            raise TypeError("Filter query must be a dictionary.")

        col = self.db[collection]
        cursor = col.find(filter_query or {}, projection)
        return str(list(cursor))


    def aggregate_documents(self, collection: str, pipeline: List[Dict]) -> List[Dict]:
        """Runs an aggregation pipeline on a MongoDB collection."""
        col = self.db[collection]
        return str(list(col.aggregate(pipeline)))

    def date_tool(self):
        today = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return today

# class Googletoolkit(Toolkit):

#     def __init__(self, credentials_path: str, sheet_id: str):
#         """Initialize Google Sheets API client and open the existing sheet."""
#         super().__init__(name="google_agent")
#         scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive",'https://www.googleapis.com/auth/spreadsheets']
#         creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
#         self.client = gspread.authorize(creds)
#         self.sheet_id = self.client.open_by_key(sheet_id)  # Open the sheet using Sheet ID
#         self.register(self.save_to_google_sheets)


#     def save_to_google_sheets(self, data: str):
#         """
#         Saves dictionary data to Google Sheets as a single row.

#         Args:
#             sheet_id: The ID of the Google Sheet.
#             sheet_range: The range where data should be appended (e.g., 'A1').
#             data: The dictionary containing the data to be saved.
#         """
#         # Convert Python dict to JSON string if it's not already a string
#         if isinstance(data, dict):  
#             data = json.dumps(data)  # Convert to JSON string
        
#         try:
#             data = json.loads(data)  # Convert JSON string to Python dict
#         except json.JSONDecodeError as e:
#             return f"Invalid JSON format: {e}"
    

#         # Ensure it's a dictionary before processing
#         if not isinstance(data, dict):
#             return "Invalid data format. Expected a dictionary."

#         # Extract values as a list
#         values = [str(data[key]) for key in sorted(data.keys())]  # Convert values to strings

#         # Authenticate with Google Sheets
#         sheet_id = "1lsPBDX-IsP7M9KxsjY-BO1hm9VvImOFqz2lzGwjAHME"
#         sheet = self.client.open_by_key(sheet_id).sheet1

#         # Check if headers exist, and add them if missing
#         existing_headers = sheet.row_values(1)
#         if not existing_headers:
#             headers = sorted(data.keys())  # Sorted to match extracted values order
#             sheet.append_row(headers)

#         # Append the extracted values
#         sheet.append_row(values)

#         return f"Data saved successfully! Google Sheet: https://docs.google.com/spreadsheets/d/{sheet_id}"

