from pymongo import MongoClient
from dotenv import load_dotenv
import os
import streamlit as st
import asyncio

load_dotenv()

uri = os.getenv("uri")
print(uri)

# Initialize MongoDB client (do this once)
try:
    client = MongoClient(uri)
    client.admin.command('ping')  # Check connection
    mongo_connected = True
    st.success("Connected to MongoDB!")
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")
    mongo_connected = False

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
from agno.models.groq import Groq
from tools.tools import MongoDBUtility, Googletoolkit
import streamlit as st
import requests

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# from pydantic import BaseModel
# from typing import Any, Dict, List, Optional, Union

# class QueryOutput(BaseModel):
#     query: Union[Dict[str, Any], List[Dict[str, Any]]]  # Supports both simple queries and aggregation pipelines
#     final_output: Union[str, int, float, List[Dict[str, Any]]]  # Supports various output types
#     dataframe: Optional[str] = None 
def run_agent():

    uri = os.getenv("uri")
    credentials_path = "credentials.json"
    sheet_id =  os.getenv("sheet_id")
    client = MongoClient(uri)

    agent = Agent(
            model= OpenAIChat(id="gpt-4o",temperature=0),
            reasoning_model=Groq(id="deepseek-r1-distill-llama-70b", temperature=0),

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
                    - Show the query that you execuited in the mongodb database
                    - Format the output in readable JSON or tabular format.
                    - Provide a **brief explanation** of the result.
                    - If no results are found, state it clearly.
                    - If `final_output` is a **list of dictionaries (documents)**, create an additional `dataframe` field:
                        - Convert the list to a Pandas DataFrame.
                        - Store the **string representation** of the DataFrame in the `dataframe` field.
                    - **Parse the string output** of the `FindDocumentsTool` and convert to list of dictionaries and save in google sheet where each document as row into Google Sheets.
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

                        5.  **Formate data problem:**
                            If the query requires retrieving records, parse it using json.loads.  Then, attempt to save the data to Google Sheets.  If saving to Google Sheets results in an error or an empty response, *immediately* convert the data into a Pandas DataFrame and output the DataFrame.
                            If the query requires retrieving records (using `FindDocumentsTool`), attempt to save the **Python list of dictionaries** directly to Google Sheets (as described in step 5). If the `save_to_google_sheets` tool returns an error or an empty/failure response, *then* convert the original **Python list of dictionaries** into a Pandas DataFrame and output the DataFrame representation in your final response.
    ```
                        **Complete Example with Error Handling**

                    **📌 Example Workflows:**

                    **Q:** "How many calls did Priya Sharma make this week?"
                    - Identify `calls` collection.
                    - Retrieve schema (ensure `caller`, `timestamp` fields exist).
                    - Construct query: `{"caller": "Priya Sharma", "timestamp": {"$gte": <7_days_ago>}}`
                    - Execute using `CountDocumentsTool`
                    - Output:   query:{
                                    "caller": "Priya Sharma",
                                    "timestamp": {"$gte": <7_days_ago>}}`
                                final output:`"Priya Sharma made 12 calls this week."`

                    **Q:** "List all failed calls in the last 7 days."
                    - Identify `calls` collection.
                    - Retrieve schema (ensure `status`, `timestamp` fields exist).
                    - Construct query: `{"status": "failed", "timestamp": {"$gte": <7_days_ago>}}`
                    - Do not use limit in the query unless specified in the question.
                    - Execute using `FindDocumentsTool`
                    - Show the tabulat formate of data as well as an output
                    - Output:  query:`{
                                "status": "failed",
                                "timestamp": {"$gte": <7_days_ago>}
                                }`
                               final output:`[{caller, receiver, timestamp}, ...]` and convert the str output from the list collections and save it in google sheet with each document in each row of google sheet or if it fails save as pandas dataframe
                               for that use pandas.Dataframe(data) and show the dataframe

                    **Q:** "What is the average call duration for completed calls?"
                    - Identify `calls` collection.
                    - Retrieve schema (ensure `duration`, `status` fields exist).
                    - Construct aggregation: `[{"$match": {"status": "completed"}}, {"$group": {"_id": None, "avg_duration": {"$avg": "$duration"}}}]`
                    - Execute using `AggregateDocumentsTool`
                    - Output: query: `[
                                        {"$match": {"status": "completed"}},
                                        {"$group": {"_id": None, "avg_duration": {"$avg": "$duration"}}}
                                        ]`
                              final output: `"The average call duration is 45.23 seconds."`

                    ---
                    Always ensure that the queries align with the **actual schema** of the collection.

                """,
                show_tool_calls=True,
                add_datetime_to_instructions=True,
                markdown=True,
    )

    return agent

def simulate_agno_response(prompt):
    # Replace this with your actual API call to Agno
    agent = run_agent()  #Calling it here it the fix so that agent will take instruction
    response = agent.print_response(prompt, stream=True)
    for char in response:
        yield char
        time.sleep(0.05)

with st.sidebar:
    if mongo_connected:
        st.header("Database Info")
        db_name = st.selectbox("Select Database", client.list_database_names())
        if db_name:
            try:
                collections = client[db_name].list_collection_names()
                st.write("Collections:", collections)
            except Exception as e:
                st.error(f"Error listing collections: {e}")
    else:
        st.write("Not connected to MongoDB.")
        st.header("Sample Questions")
    sample_question = st.selectbox(
        "Choose a question:",
        [
            "How many calls did Priya Sharma make this week?",
            "List all failed calls in the last 7 days.",
            "What is the average call duration for completed calls?",
            "Show me all call records",
        ],
    )

# Main App
st.title("AI Copilot to Analyze your data")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about your data"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Agent response

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        agent = run_agent()
        # Generate response with error handling
        with st.spinner("Analyzing your question..."):
            result = agent.run(prompt)


        # Option 1: Get the formatted string representation of each tool call
        formatted_calls = result.formatted_tool_calls
        print("Formatted Tool Calls:")
        for call in formatted_calls:
            print(f"- {call}")
        # Option 2: Get the names of the tools executed
        # This approach extracts the unique names of the tools that were invoked.
        tool_interactions = result.tools
        tool_names_used = set() # Use a set to store unique names
        if tool_interactions:
            for interaction in tool_interactions:
                tool_names_used.add(interaction.get('tool_name')) # Safely get tool_name
        print("\nUnique Tool Names Used:")
        if tool_names_used:
            for name in tool_names_used:
                print(f"- {name}")
        else:
            print("- No tools were used.")
        # Option 3: Get detailed information for each tool interaction
        print("\nDetailed Tool Interactions:")
        if tool_interactions:
            for i, interaction in enumerate(tool_interactions):
                print(f"Interaction {i+1}:")
                print(f"  Name: {interaction.get('tool_name')}")
                print(f"  Args: {interaction.get('tool_args')}")
                # print(f"  Result: {interaction.get('content')}") # Result can be long, print if needed
                print("-" * 10)
        else:
            print("- No tool interactions occurred.")
        # --- START: Display Tool Interactions ---
        if result.tools: # Check if any tools were used
            with st.expander("🔍 Show Tool Interactions"):
                st.markdown("---") # Separator
                for i, interaction in enumerate(result.tools):
                    tool_name = interaction.get('tool_name', 'N/A')
                    tool_args = interaction.get('tool_args', {})

                    # Format arguments nicely (convert dict to formatted string)
                    import json
                    try:
                        args_str = json.dumps(tool_args, indent=2)
                    except TypeError: # Handle non-serializable args if necessary
                        args_str = str(tool_args)

                    st.markdown(f"**Interaction {i+1}: {tool_name}**")
                    st.code(args_str, language='json')
                    st.markdown("---") # Separator between interactions
        print(result)
        # Display assistant response
        st.markdown(result.content)
        
        st.session_state.messages.append({"role": "assistant", "content":result.content })




