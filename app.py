from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

uri = os.getenv("uri")
print(uri)

try:
    client = MongoClient(uri)
    # The ismaster command is used to check the connection
    server_status = client.admin.command("ismaster")
    print("MongoDB Connection Successful ✅")
except Exception as e:
    print("MongoDB Connection Failed ❌", e)

# List all databases
print("Databases:", client.list_database_names())

from pymongo.mongo_client import MongoClient
import openai
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import pandas as pd
from dotenv import load_dotenv
import os
from textwrap import dedent
from agno.tools.thinking import ThinkingTools
from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools.pandas import PandasTools
from tools.tools import MongoDBUtility, Googletoolkit

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def run_agent():

    uri = os.getenv("uri")
    credentials_path = os.getenv("credentials_path")
    sheet_id =  os.getenv("sheet_id")
    client = MongoClient(uri)

    agent = Agent(
            model= OpenAIChat(id="gpt-4o"),
            tools = [MongoDBUtility(),ThinkingTools(),Googletoolkit(credentials_path, sheet_id),PandasTools()],
            instructions="""
                    You are an intelligent MongoDB assistant that dynamically constructs and executes queries based on user input. Follow these steps:
                    You are a MongoDB assistant that **must** execute the query using the appropriate tools.

                    1️ **Identify the Relevant Collection:**
                    - Use `ListCollectionsTool` to retrieve the available collections.
                    - Match the collection name with the user query if there is a some match in name from the user input.
                    - If unsure, ask the user for clarification.

                    2️ **Retrieve Schema Information:**
                    - Use `MongoSchemaTool` to get the schema of the identified collection.
                    - Extract the correct field names and data types.

                    3️ **Generate an Optimized Query:**
                    - Construct a MongoDB query based on the user's intent.
                    - Use appropriate filters (`$eq`, `$gte`, `$lte`, `$regex`, etc.).
                    - **Only include necessary fields** in the projection, avoiding `_id` unless required.
                    - If the query involves aggregation (e.g., counting or averaging), use `AggregateDocumentsTool`.

                    4️ **Execute the Query:**
                    - If the query is a **count operation**, use `CountDocumentsTool`.
                    - If retrieving multiple documents, use `FindDocumentsTool`.
                    - For summary statistics (e.g., average duration), use `AggregateDocumentsTool`.

                    5️ **Return a Clear and Concise Response:**
                    - Format the output in readable JSON or tabular format.
                    - Provide a **brief explanation** of the result.
                    - If no results are found, state it clearly.
                    - **Parse the string output** of the `FindDocumentsTool` with `json.loads()`Save the parsed list of dictionaries to Google Sheets.
                    

                    - save to google sheets if the question is to get the records form the mongodb database

                    ---
                    3. **Error Handling and Fallback:**
                        - **Problem:** The agent currently doesn't gracefully handle errors when saving to Google Sheets.
                        - **Solution:** Implement error handling with the Pandas DataFrame fallback *in the agent's logic*.   Because you can't directly modify the tool's code from inside the agent instructions, the agent needs to *react* to a failed Google Sheets save.

                        Here's the general pattern for agent instruction:

                        ```
                        If the query requires retrieving records, parse it using json.loads.  Then, attempt to save the data to Google Sheets.  If saving to Google Sheets results in an error or an empty response, *immediately* convert the data into a Pandas DataFrame and output the DataFrame.
                        ```

                        4.  **Google Sheets API Permissions:**
                        -   **Problem:** Incorrect or missing permissions on the service account used by `gspread`.
                        -   **Solution:**

                            *   **Check the Credentials:**  Ensure the `credentials.json` file is valid and contains the correct service account credentials.
                            *   **Verify Sheet Sharing:**  The service account's email address MUST be explicitly granted "Editor" access to the Google Sheet. Sharing the sheet with your personal Google account is NOT sufficient.

                        5. **String Conversion:**
                        - **Problem:** The `save_to_google_sheets` method converts all data to strings before saving. If your data contains numeric or boolean values, they will be treated as strings in Google Sheets.
                        - **Solution:**  If you need to preserve data types, consider modifying your MongoDB queries to return data in the desired formats, or add logic to the `save_to_google_sheets` function to handle different data types.  However, for simplicity, treating everything as a string is often sufficient.

                        **Complete Example with Error Handling**

                    **📌 Example Workflows:**

                    **Q:** "How many calls did Priya Sharma make this week?"
                    - Identify `calls` collection.
                    - Retrieve schema (ensure `caller`, `timestamp` fields exist).
                    - Construct query: `{"caller": "Priya Sharma", "timestamp": {"$gte": <7_days_ago>}}`
                    - Execute using `CountDocumentsTool`
                    - Output: `"Priya Sharma made 12 calls this week."`

                    **Q:** "List all failed calls in the last 7 days."
                    - Identify `calls` collection.
                    - Retrieve schema (ensure `status`, `timestamp` fields exist).
                    - Construct query: `{"status": "failed", "timestamp": {"$gte": <7_days_ago>}}`
                    - Do not use limit in the query unless specified in the question.
                    - Execute using `FindDocumentsTool`
                    - Show the tabulat formate of data as well as an output
                    - Output: `[{caller, receiver, timestamp}, ...]` and convert the str output from the list collections and save it in google sheet with the question or if it fails save as pandas dataframe
                        for that use pandas.Dataframe(data) and show the dataframe

                    **Q:** "What is the average call duration for completed calls?"
                    - Identify `calls` collection.
                    - Retrieve schema (ensure `duration`, `status` fields exist).
                    - Construct aggregation: `[{"$match": {"status": "completed"}}, {"$group": {"_id": None, "avg_duration": {"$avg": "$duration"}}}]`
                    - Execute using `AggregateDocumentsTool`
                    - Output: `"The average call duration is 45.23 seconds."`

                    ---
                    Always ensure that the queries align with the **actual schema** of the collection.

                """,
                show_tool_calls=True,

                markdown=True,
    )

    agent.print_response("List all failed calls in the last 7 days.", stream=True)

run_agent()