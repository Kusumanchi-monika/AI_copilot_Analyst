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

# --- Pricing (per 1 million tokens) ---
# Please verify current pricing from OpenAI and Groq documentation
MODEL_PRICING_PER_MILLION_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 15.00},
    # Assuming Groq is currently free or has negligible cost for this example model
    "deepseek-r1-distill-llama-70b": {"input": 0.0, "output": 0.0},
    # Add other models here if you use them
}

# Define the ID of the primary model whose pricing will be used for estimation
# If usage details aren't broken down per model in the response.
PRIMARY_MODEL_ID_FOR_COST = "gpt-4o"
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
                    - If retrieving multiple documents **and the user asks to list or save them**, use `FindDocumentsTool`.
                    - For summary statistics (e.g., average duration), use `AggregateDocumentsTool`.

                    4.5 **Handling `FindDocumentsTool` Results for Listing/Saving:**
                        - **IF** you just successfully executed `FindDocumentsTool` because the user asked to list or see multiple records, **AND** that tool returned a list of documents (let's call the variable containing this list `retrieved_data_list`):
                            - **THEN:** You **MUST** proceed to call the `save_list_to_csv` tool immediately in the next step.
                            - **WHEN CALLING `save_list_to_csv`:**
                                - You **MUST** provide the `retrieved_data_list` (the actual list object returned by `FindDocumentsTool`) as the value for the `data` argument. **Do NOT forget this argument.**
                                - **Example:** If `FindDocumentsTool` returned `[{'field1': 'val1'}, {'field1': 'val2'}]`, the call should look like `save_list_to_csv(data=[{'field1': 'val1'}, {'field1': 'val2'}], output_filename='optional_name.csv')`.
                                - **Filename:** If the user suggested a filename (e.g., "save as my_report.csv"), pass it to the `output_filename` argument. If no filename was mentioned, **do not include the `output_filename` argument at all** (let the tool generate one automatically).
                            - **Do NOT try to convert the `retrieved_data_list` to a string or summarize it before passing it to `save_list_to_csv`. Pass the raw list object.**

                    5️ **Return a Clear and Concise Response:**
                        - Show the query that you executed in the MongoDB database (if applicable, like for Find, Count, Aggregate).
                        - **If data was saved to a file using `save_list_to_csv`**, **include the exact confirmation message string returned by that tool** in your final response to the user (e.g., "Successfully generated CSV data (50 records) named 'output_20240401_123456.csv'. The user can now download it.").
                        - If `FindDocumentsTool` was used *without* saving (e.g., user asked "show me the first 5 failed calls"), present a small sample of the results directly in the response (e.g., as a markdown table or formatted list), but avoid showing excessively large lists. Mention how many total records were found if relevant.
                        - For counts or single-value aggregations, provide a brief explanation of the result.

                    ---
                    **📌 Example Workflow (Find and Save):**

                    **Q:** "List all failed calls in the last 7 days and save them."
                    1. Identify `call_logs_sample` collection.
                    2. Get schema (confirm `call_status`, `call_time`).
                    3. Construct filter: `{"call_status": "failed", "call_time": {"$gte": <7_days_ago>}}`
                    4. **Execute `FindDocumentsTool`** with the filter. Assume it returns a list: `failed_calls_list = [{'agent': 'A', ...}, {'agent': 'B', ...}]`
                    5. **Immediately execute `save_list_to_csv`**, **passing the list from step 4**: `save_list_to_csv(data=failed_calls_list)` (no filename specified by user). Assume it returns the string: `"Successfully generated CSV data (15 records) named 'output_20240401_180000.csv'. The user can now download it."`
                    6. **Final Output to User:**
                    ```
                    **Query Executed (Find):**
                    ```json
                    {
                        "call_status": "failed",
                        "call_time": {"$gte": "2025-03-25"}
                    }
                    ```

                    **Result:**
                    Successfully generated CSV data (15 records) named 'output_20240401_180000.csv'. The user can now download it.
                    ```

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
        agent = run_agent()

        # Clear previous download state
        if 'last_csv_download' in st.session_state:
            del st.session_state['last_csv_download']

        # Initialize variables for usage and cost
        result = None
        agent_ran_successfully = False
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        estimated_cost = 0.0

        with st.spinner("Analyzing your question and fetching data..."):
            try:
                result = agent.run(prompt)
                agent_ran_successfully = True
                print("Agent RunResponse:", result) # Add print statement for debugging

                # --- Extract Token Usage ---
                # Try extracting from the metrics of the LAST message first
                if (result and hasattr(result, 'messages') and result.messages and
                    hasattr(result.messages[-1], 'metrics') and result.messages[-1].metrics):

                    last_msg_metrics = result.messages[-1].metrics
                    # Use attribute access assuming it's an object like MessageMetrics
                    prompt_tokens = getattr(last_msg_metrics, 'prompt_tokens', 0)
                    completion_tokens = getattr(last_msg_metrics, 'completion_tokens', 0)
                    total_tokens = getattr(last_msg_metrics, 'total_tokens', prompt_tokens + completion_tokens)

                    # If total_tokens is still 0, maybe prompt/completion weren't set correctly, try total directly
                    if total_tokens == 0 and hasattr(last_msg_metrics, 'total_tokens'):
                         total_tokens = getattr(last_msg_metrics, 'total_tokens', 0)
                         # Cannot accurately split into prompt/completion if only total is available here

                # --- Fallback: Try extracting from top-level 'metrics' dictionary ---
                elif (result and hasattr(result, 'metrics') and isinstance(result.metrics, dict)):
                     metrics_dict = result.metrics
                     if ('prompt_tokens' in metrics_dict and isinstance(metrics_dict['prompt_tokens'], list) and metrics_dict['prompt_tokens']):
                         prompt_tokens = metrics_dict['prompt_tokens'][-1]
                     if ('completion_tokens' in metrics_dict and isinstance(metrics_dict['completion_tokens'], list) and metrics_dict['completion_tokens']):
                         completion_tokens = metrics_dict['completion_tokens'][-1]
                     if ('total_tokens' in metrics_dict and isinstance(metrics_dict['total_tokens'], list) and metrics_dict['total_tokens']):
                         total_tokens = metrics_dict['total_tokens'][-1]
                     else:
                          total_tokens = prompt_tokens + completion_tokens # Calculate if not present


                # --- Calculate Estimated Cost ---
                if total_tokens > 0 and PRIMARY_MODEL_ID_FOR_COST in MODEL_PRICING_PER_MILLION_TOKENS:
                    pricing = MODEL_PRICING_PER_MILLION_TOKENS[PRIMARY_MODEL_ID_FOR_COST]
                    # Ensure we have valid numbers before calculation
                    prompt_tokens_calc = prompt_tokens if isinstance(prompt_tokens, (int, float)) else 0
                    completion_tokens_calc = completion_tokens if isinstance(completion_tokens, (int, float)) else 0

                    input_cost = (prompt_tokens_calc / 1_000_000) * pricing.get('input', 0)
                    output_cost = (completion_tokens_calc / 1_000_000) * pricing.get('output', 0)
                    estimated_cost = input_cost + output_cost
                elif total_tokens > 0:
                     st.warning(f"Pricing not found for primary model '{PRIMARY_MODEL_ID_FOR_COST}'. Cost cannot be estimated accurately.")


            except Exception as e:
                st.error(f"An error occurred while running the agent or processing results: {e}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")


        # --- Display Results, Download Button, and Usage Info ---
        if agent_ran_successfully and result:
            # Display Tool Interactions (Optional Expander)
            # ... (tool display code remains the same) ...
            if result.tools:
                 with st.expander("🔍 Show Tool Interactions"):
                      st.markdown("---") # Start of tool interactions display
                      for i, interaction in enumerate(result.tools):
                           tool_name = interaction.get('tool_name', 'N/A')
                           tool_args = interaction.get('tool_args', {})
                           tool_content_str = str(interaction.get('content', 'N/A')) # Content is now string

                           try:
                                args_str = json.dumps(tool_args, indent=2)
                           except TypeError:
                                args_str = str(tool_args)

                           st.markdown(f"**Interaction {i+1}: {tool_name}**")
                           st.code(args_str, language='json')
                           st.write("**Result (Sent to LLM):**")
                           st.caption(tool_content_str) # Display the string result
                           st.markdown("---") # Separator between interactions


            # Display Main Assistant Response
            assistant_response_content = result.content if result.content else "I couldn't generate a response based on the tool interactions."
            st.markdown(assistant_response_content)

            # Store Message
            current_message = {"role": "assistant", "content": assistant_response_content}
            st.session_state.messages.append(current_message)


            # Check session_state for CSV Download Button
            # ... (download button code remains the same) ...
            if 'last_csv_download' in st.session_state:
                 download_info = st.session_state['last_csv_download']
                 try:
                     st.download_button(
                         label=f"📥 Download {download_info['filename']}",
                         data=download_info['csv_content'].encode('utf-8'),
                         file_name=download_info['filename'],
                         mime='text/csv',
                         key=f"download_{download_info['filename']}_{len(st.session_state.messages)}"
                     )
                 except Exception as btn_e:
                      st.error(f"Error creating download button: {btn_e}")


            # --- Display Token Usage and Cost ---
            if total_tokens > 0:
                cost_string = f"${estimated_cost:.6f}" # Format cost
                # Display the extracted token counts
                usage_info = f"Tokens Used: {prompt_tokens} Prompt + {completion_tokens} Completion = {total_tokens} Total | Estimated Cost: {cost_string} (based on {PRIMARY_MODEL_ID_FOR_COST})"
                st.caption(usage_info)
            elif agent_ran_successfully: # Only show 'not available' if agent ran but no tokens found
                st.caption("Token usage information not available for this run.")


        elif not agent_ran_successfully:
            # Handle case where agent.run() itself failed
             error_message = "Sorry, I encountered an error and couldn't process your request."
             st.markdown(error_message)
             st.session_state.messages.append({"role": "assistant", "content": error_message})