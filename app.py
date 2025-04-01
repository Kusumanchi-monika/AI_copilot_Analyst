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
from tools.tools import MongoDBUtility
import streamlit as st
import requests

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

import streamlit as st # Make sure streamlit is imported here
import csv
import json
from io import StringIO
from datetime import datetime
from typing import List, Dict, Any, Optional

# Assuming ObjectId might be present from MongoDB results
try:
    from bson import ObjectId
except ImportError:
    ObjectId = type(None)

# No longer need TypedDict for return type, as it returns str
# class CsvSaveResult(TypedDict): ... # Remove this

def save_list_to_csv(
    data: List[Dict[str, Any]],
    output_filename: Optional[str] = None
) -> str: # <--- Changed return type hint to str
    """
    Generates CSV content from a list of dictionaries, stores it in
    st.session_state['last_csv_download'] for frontend download,
    and returns only a confirmation message string suitable for the LLM.

    Args:
        data (List[Dict[str, Any]]): List of data records to convert.
        output_filename (Optional[str]): Desired output filename. Generated if None.

    Returns:
        str: A confirmation or error message string.
    """
    # --- Clear any previous download state ---
    if 'last_csv_download' in st.session_state:
        try:
            del st.session_state['last_csv_download']
        except Exception: # Handle potential errors if deletion fails unexpectedly
             pass # Log this if needed

    if not data:
        return "Error: No data provided to generate CSV." # Return simple string

    # --- Generate filename ---
    is_generated_filename = False
    if not output_filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"output_{timestamp}.csv"
        is_generated_filename = True
    elif not output_filename.lower().endswith('.csv'):
        output_filename += '.csv'

    csv_content_string = None
    message_for_llm = f"Error: Could not generate CSV for {output_filename}." # Default error

    try:
        # --- Data Preprocessing ---
        processed_data = []
        all_keys = set()
        for row_dict in data:
            processed_row = {}
            for k, v in row_dict.items():
                key_str = str(k)
                all_keys.add(key_str)
                if isinstance(v, (ObjectId, datetime)):
                    processed_row[key_str] = str(v)
                elif isinstance(v, (dict, list)):
                    processed_row[key_str] = json.dumps(v) # Ensure complex types are stringified
                elif v is None:
                    processed_row[key_str] = "" # Represent None as empty string in CSV
                else:
                    processed_row[key_str] = v
            processed_data.append(processed_row)

        # Determine fieldnames consistently
        if not all_keys and processed_data: # Handle case where first row might dictate keys
             all_keys = set(processed_data[0].keys())
        fieldnames = sorted(list(all_keys))


        # --- Generate CSV String using StringIO ---
        string_io = StringIO()
        # Use quoting=csv.QUOTE_MINIMAL or csv.QUOTE_NONNUMERIC based on needs
        writer = csv.DictWriter(string_io, fieldnames=fieldnames, extrasaction='ignore', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(processed_data)
        csv_content_string = string_io.getvalue()
        string_io.close()

        # --- Prepare success message and store data ---
        message_for_llm = f"Successfully generated CSV data ({len(data)} records) named '{output_filename}'. The user can now download it."
        if is_generated_filename:
             message_for_llm += " (Filename was auto-generated)"

        # --- Store details in session_state for Streamlit frontend ---
        st.session_state['last_csv_download'] = {
            "filename": output_filename,
            "csv_content": csv_content_string,
            "message": message_for_llm # Store the message too, might be useful for context
        }

        return message_for_llm # <--- Return ONLY the confirmation string

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during CSV generation: {e}\n{error_details}") # Log detailed error
        message_for_llm = f"Error generating CSV data for {output_filename}: {e}"
        # Ensure state is cleared on error
        if 'last_csv_download' in st.session_state:
            try:
                del st.session_state['last_csv_download']
            except Exception:
                pass
        return message_for_llm # <--- Return only the error string

# from pydantic import BaseModel
# from typing import Any, Dict, List, Optional, Union

# class QueryOutput(BaseModel):
#     query: Union[Dict[str, Any], List[Dict[str, Any]]]  # Supports both simple queries and aggregation pipelines
#     final_output: Union[str, int, float, List[Dict[str, Any]]]  # Supports various output types
#     dataframe: Optional[str] = None 
def run_agent():

    uri = os.getenv("uri")
    sheet_id =  os.getenv("sheet_id")
    client = MongoClient(uri)

    agent = Agent(
            model= OpenAIChat(id="gpt-4o",temperature=0),
            # model=Claude(id="claude-3-7-sonnet-20250219", temperature=0),

            reasoning_model=Groq(id="deepseek-r1-distill-llama-70b", temperature=0),

            tools = [MongoDBUtility(),ThinkingTools(),PandasTools(), save_list_to_csv],
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
                    - If retrieving multiple documents, use `FindDocumentsTool`and save in csv file using the tool `save_list_to_csv` ".
                    - For summary statistics (e.g., average duration), use `AggregateDocumentsTool`.
                

                    4.5 **Save Results if  the user says list the records or what are the records or Listing/Saving Records:**
                        - **If** the user's request implies listing or saving multiple records **and** you just used `FindDocumentsTool` which returned a list of documents (let's call this list `retrieved_data`):
                            - **Determine the desired format:** Default to CSV unless the user explicitly asks for JSON (if you have a JSON save tool).
                            - **Determine the filename:** If the user suggests a filename (e.g., "save it as failed_emails.csv"), use that. Otherwise, do not provide a filename to the tool (let it generate one).
                            - **If saving as CSV:**
                                - **Call the `save_list_to_csv` tool.**
                -                - Pass the `retrieved_data` (the actual Python list object from `FindDocumentsTool`) to the `data_to_save` argument.
                +                - Pass the `retrieved_data` (the actual Python list object from `FindDocumentsTool`) to the `data` argument.
                                - If a specific filename was requested by the user, pass it to the `output_filename` argument. Otherwise, pass nothing for `output_filename`.
                            # - (Add similar logic here if you also have a JSON saving tool)
                            - **Crucially: Do not convert the `retrieved_data` list to a string** before passing it to the saving tool.

                5️ **Return a Clear and Concise Response:**
                    - Show the query that you execuited in the mongodb database
            -        - **If you saved data to a file (CSV or JSON), clearly state the filename and format** in your response, using the confirmation message returned by the saving tool.
            +        - **If data was saved to a file, include the exact confirmation message returned by the saving tool** (e.g., "Successfully saved X records to Y.csv") in your final response to the user.
                    - Format the output in readable JSON or tabular format for the user's view.
                    - Provide a **brief explanation** of the result.


                    ---
                    3. **Error Handling and Fallback:**
                        - **Problem:** The agent currently doesn't gracefully handle errors when saving to Google Sheets.
                        - **Solution:** Implement error handling with the Pandas DataFrame fallback *in the agent's logic*.   Because you can't directly modify the tool's code from inside the agent instructions, the agent needs to *react* to a failed Google Sheets save.

                        Here's the general pattern for agent instruction:


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
        # --- IMPORTANT: Clear previous download state BEFORE running agent ---
        # Ensures no stale button appears if the current run doesn't generate a file
        if 'last_csv_download' in st.session_state:
            del st.session_state['last_csv_download']

        # Generate response with error handling
        result = None
        agent_ran_successfully = False
        with st.spinner("Analyzing your question..."):
            try:
                result = agent.run(prompt) # This might take time
                agent_ran_successfully = True
            except Exception as e:
                st.error(f"An error occurred while running the agent: {e}")
                # Optionally log the full traceback for debugging
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")

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
        if agent_ran_successfully and result:

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
                        st.write("**Result (Sent to LLM):**")
                        # st.caption(tool_content_str) # Display the string result
                        st.markdown("---")
            # Display Main Assistant Response
            assistant_response_content = result.content if result.content else "I couldn't generate a response based on the tool interactions."
            st.markdown(assistant_response_content)

            # Store Message *before* adding the download button
            current_message = {"role": "assistant", "content": assistant_response_content}
            st.session_state.messages.append(current_message)

            # --- Check session_state for CSV Download Button AFTER agent run ---
            if 'last_csv_download' in st.session_state:
                download_info = st.session_state['last_csv_download']
                try:
                    st.download_button(
                        label=f"📥 Download {download_info['filename']}",
                        data=download_info['csv_content'].encode('utf-8'), # Encode string to bytes
                        file_name=download_info['filename'],
                        mime='text/csv',
                        # Use a more robust key if needed, combining filename and timestamp or message index
                        key=f"download_{download_info['filename']}_{len(st.session_state.messages)}"
                    )
                    # Optional: Add download info to the stored message for potential re-rendering later
                    # current_message['download_info'] = download_info
                except KeyError as ke:
                    st.error(f"Failed to create download button. Missing data: {ke}")
                except Exception as btn_e:
                    st.error(f"Error creating download button: {btn_e}")

                # Decide whether to clear 'last_csv_download' here or rely on the clear at the start of the next turn.
                # Clearing at the start is generally safer.

        elif not agent_ran_successfully:
            # Handle case where agent.run() itself failed
                error_message = "Sorry, I encountered an error and couldn't process your request."
                st.markdown(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
