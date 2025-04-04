# AI Copilot Analyst for MongoDB

## Overview

This project is an interactive AI-powered data analyst assistant built with Streamlit. It allows users to query and analyze data stored in a MongoDB database using natural language questions. The backend leverages the Agno agent framework, orchestrating Large Language Models (LLMs) like OpenAI's GPT-4o and Groq's Llama models, along with custom tools, to dynamically generate and execute MongoDB queries.

Users can ask questions like "How many calls failed last week?", "List all appointments for John Doe and save them to CSV", or "What's the average email length?". The application displays the results, the query used, provides options to download data (if applicable), and shows estimated token usage and cost for the interaction.

## Features

*   **Natural Language Querying:** Interact with your MongoDB data using plain English.
*   **MongoDB Integration:** Connects securely to your MongoDB Atlas cluster (or local instance).
*   **Dynamic Query Generation:** Automatically constructs appropriate MongoDB `find`, `count`, or `aggregate` queries based on user intent.
*   **Custom Tool Usage:** Employs specific tools for:
    *   Listing MongoDB collections.
    *   Fetching collection schemas.
    *   Executing `find`, `count`, and `aggregate` operations.
    *   Saving query results to CSV format.
*   **CSV Data Export:** Provides a download button for results fetched using `FindDocumentsTool` when requested.
*   **Interactive Chat Interface:** Built with Streamlit for a user-friendly experience.
*   **Tool Interaction Visibility:** Optionally displays the sequence of tools called by the agent and their arguments/results for transparency.
*   **Token Usage & Cost Estimation:** Shows the number of prompt/completion tokens used and estimates the cost for the LLM interaction (based on configured pricing).
*   **LLM Integration:** Uses Agno to manage interactions with powerful models like GPT-4o (for main task) and Groq models (potentially for reasoning).

## Technology Stack

*   **Frontend:** Streamlit
*   **Backend:** Python
*   **AI Agent Framework:** Agno (`agno-py`)
*   **LLMs:**
    *   OpenAI API (e.g., GPT-4o)
    *   Groq API (e.g., deepseek-r1-distill-llama-70b)
*   **Database:** MongoDB (via `pymongo`)
*   **Configuration:** `python-dotenv`
*   **Data Handling:** `pandas` (optional, via PandasTools if used), `csv`

## Architecture / How it Works

1.  **User Input:** The user types a natural language question into the Streamlit chat interface.
2.  **Streamlit Backend:** Receives the input and passes it to the Agno agent.
3.  **Agno Agent:**
    *   Receives the user prompt and system instructions.
    *   Uses a reasoning LLM (e.g., Groq) to plan the steps needed.
    *   Uses a primary LLM (e.g., GPT-4o) to select and invoke appropriate tools based on the plan and instructions.
    *   **Tool Execution:**
        *   Calls tools like `ListCollectionsTool` or `MongoSchemaTool` to understand the database structure.
        *   Constructs MongoDB queries (filters, aggregations).
        *   Calls tools like `CountDocumentsTool`, `FindDocumentsTool`, or `AggregateDocumentsTool` to execute queries against the MongoDB database via `pymongo`.
        *   If `FindDocumentsTool` results need saving, it calls the `save_list_to_csv` tool.
    *   Receives results back from the tools.
4.  **Response Generation:** The primary LLM synthesizes the results from tool calls into a coherent, user-friendly response.
5.  **Streamlit Frontend:**
    *   Displays the agent's final text response.
    *   If the `save_list_to_csv` tool stored data in `st.session_state`, a `st.download_button` is rendered.
    *   Displays token usage and estimated cost information retrieved from the agent's run response.
    *   Optionally displays tool interactions in an expander.

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Kusumanchi-monika/AI_copilot_Analyst
    cd AI_copilot_Analyst
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Make sure you have a `requirements.txt` file listing all necessary packages (like `streamlit`, `pymongo`, `openai`, `groq`, `agno-py`, `python-dotenv`, `pandas` etc.).
    ```bash
    pip install -r requirements.txt
    ```
    *If you don't have `requirements.txt`, you can generate it after installing packages manually:*
    ```bash
    # Install packages manually first if needed:
    # pip install streamlit pymongo openai groq agno-py python-dotenv pandas "pymongo[srv]" # Add others as needed
    # Then generate the file:
    # pip freeze > requirements.txt
    ```

4.  **Configure Environment Variables:**
    Create a file named `.env` in the root directory of the project and add your credentials:
    ```dotenv
    # .env file
    OPENAI_API_KEY="your_openai_api_key_here"
    GROQ_API_KEY="your_groq_api_key_here"
    # Your MongoDB connection string (ensure it includes the database name if needed by your tools)
    uri="mongodb+srv://<username>:<password>@<your-cluster-url>/?retryWrites=true&w=majority&appName=<YourAppName>"
    # Optional: Google Sheet ID if using Google Sheets tools (not shown in current code but for completeness)
    # sheet_id="your_google_sheet_id"
    ```
    **Important:** Add `.env` to your `.gitignore` file to avoid committing secrets.

## Running the Application

Ensure your virtual environment is activated. Run the Streamlit app from your terminal:

```bash
streamlit run app.py

```
[streamlit-app-2025-04-01-23-04-10.webm](https://github.com/user-attachments/assets/c93808bb-9005-4d3f-a6ef-e8602c97ea95)

