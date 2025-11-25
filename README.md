# Personal Assistant Agent POC

This project is a proof-of-concept for a personal assistant agent with a Text-based User Interface (TUI). It uses LangChain and LangGraph for the agent logic, Google's Gemini Pro as the language model, and Textual for the TUI.

## Setup

This project uses `uv` for package management. A `Makefile` is provided to simplify common tasks.

### 1. Initial Setup

To set up the project and install all dependencies:

```sh
make install
```
This command will:
*   Create a Python virtual environment (`.venv`).
*   Ensure `uv` is installed globally.
*   Install all project dependencies (including `langchain`, `langgraph`, `textual`, etc.) into the virtual environment.

### 2. Configure API Keys

The agent requires several API keys and credentials:

#### a) Google API Key (for Gemini Models)

1.  Open the `.env` file.
2.  Replace `"YOUR_GOOGLE_API_KEY"` with your actual [Google API key](https://ai.google.dev/).

    ```
    GOOGLE_API_KEY="AIzaSy..."
    ```

#### b) Google Sheets OAuth 2.0 Credentials

To allow the agent to access Google Sheets on your behalf, you need to provide OAuth 2.0 credentials.

1.  **Enable the APIs:** Go to the [Google Cloud Console](https://console.cloud.google.com/) and make sure you have a project selected.
    *   Enable the **Google Sheets API**.
    *   Enable the **Google Drive API** (the Sheets API uses this to create new files).

2.  **Create OAuth Credentials:**
    *   Go to the [APIs & Services > Credentials](https://console.cloud.google.com/apis/credentials) page.
    *   Click **+ CREATE CREDENTIALS** and select **OAuth client ID**.
    *   If prompted, configure the consent screen. For an internal app, you can keep it simple. Select **Desktop app** as the application type.
    *   Give it a name (e.g., "Michaelmas CLI").
    *   Click **Create**.

3.  **Download Credentials File:**
    *   After creating the client ID, a modal will appear. Click **DOWNLOAD JSON**.
    *   Rename the downloaded file to `credentials.json` and place it in the root of this project directory. The file's contents should look similar to the placeholder that was created.

#### c) Tavily API Key (for Web Search)

To enable web search, you need a free API key from Tavily.

1.  **Sign up:** Go to [https://tavily.com/](https://tavily.com/) and create a free account.
2.  **Get API Key:** Once logged in, you can find your API key on your dashboard.
3.  **Update `.env`:** Open your `.env` file and replace `"YOUR_TAVILY_API_KEY"` with your actual Tavily API key.

    ```
    TAVILY_API_KEY="tvly-xxxxxxxxxxxxxxxxxxxxxxxxx"
    ```

### 3. Run the Application & Authenticate

To launch the Michaelmas TUI application:

```sh
make run
```

**First Run Only (for Google Sheets):** The first time you ask the agent to do something with Google Sheets (e.g., "create a new spreadsheet called 'My Test'"), it will automatically open a new tab in your web browser, asking you to authorize access to your Google account. After you approve, it will save a `token.json` file in the project directory so you don't have to log in every time.

## Commands

The TUI supports special slash commands for meta-actions:

-   `/list_models`: Lists all available Gemini and local Ollama models that support content generation. This is useful for debugging and for finding a model name to use with the `/set_model` command.
-   `/set_model <model_name>`: Changes the AI model used for the agent in the current session.
    -   Example for Gemini: `/set_model gemini-1.5-flash-latest`
    -   Example for Ollama: `/set_model ollama:llama3`

### Maintenance Commands

-   `make clean`: Removes the virtual environment, log files, conversation history, and temporary credential tokens. Useful for a fresh start.
