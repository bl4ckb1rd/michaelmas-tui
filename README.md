# Michaelmas: Advanced AI Personal Assistant

**Michaelmas** is a sophisticated, terminal-based AI assistant orchestrated to optimize the digital life, workflows, and information streams of the modern IT professional. Built with Python, LangGraph, and Textual, it offers a seamless TUI experience.

## Key Features

*   **Multi-Model Support**: Switch dynamically between **Google Gemini** (Cloud) and **Ollama** (Local) models.
*   **Agentic Capabilities**:
    *   **General Chat**: Intelligent conversation with context awareness.
    *   **Web Research**: Autonomous web searching using **Tavily** to find real-time information.
    *   **Google Sheets Management**: Create, read, and update Google Sheets directly from the chat.
*   **Advanced TUI**:
    *   **Tool Management**: Enable/disable specific tools or all tools globally via the settings menu.
    *   **Streaming Responses**: Watch the AI think and type in real-time.
    *   **Conversations Sidebar**: Automatically saves your chat history. Resume any conversation instantly.
    *   **Status Bar**: Real-time tracking of token usage and costs (Session & Monthly).
    *   **Command History**: Use `Ctrl+Up`/`Ctrl+Down` to cycle through your previous inputs.
    *   **Multi-line Input**: Paste large blocks of code or text easily.

## Setup

This project uses `uv` for fast and reliable package management. A `Makefile` is provided for convenience.

### 1. Initial Setup

Clone the repository and run the install command:

```sh
make install
```
This uses `uv` to create a virtual environment and install all dependencies defined in `pyproject.toml`.

### 2. Configure API Keys

Create a `.env` file in the root directory. You can copy `.env.example` if provided, or simply create it.

#### Required Keys:

**a) Google API Key (for Gemini Models)**
Get it from [Google AI Studio](https://aistudio.google.com/).
```env
GOOGLE_API_KEY="AIzaSy..."
```

**b) Tavily API Key (for Web Search)**
Get a free key from [tavily.com](https://tavily.com/).
```env
TAVILY_API_KEY="tvly-..."
```

**c) OpenAI API Key (Optional)**
If you want to use OpenAI models.
```env
OPENAI_API_KEY="sk-..."
```

#### Google Sheets Setup (OAuth 2.0)

To manage Sheets, you need a `credentials.json` file:
1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  Enable the **Google Sheets API** and **Google Drive API**.
3.  Create **OAuth Client ID** credentials (Desktop App).
4.  Download the JSON file, rename it to `credentials.json`, and place it in the project root.

### 3. Run the Application

Launch the TUI:

```sh
make run
```

*On the first run using Google Sheets, a browser window will open for you to authorize access.*

## Usage

### Chat & Commands

Type naturally to chat. Use slash commands to control the system:

*   `/help`: Show the list of available commands.
*   `/new`: Start a fresh conversation.
*   `/settings`: Configure LLM settings (e.g., temperature, max output tokens, and tool availability).
*   `/model`: Open the model selector to switch AI models.
*   `/list_models`: List all available Gemini and local Ollama models (use `/model` to select).

### Keyboard Shortcuts

*   **Enter**: Send message.
*   **Shift+Enter**: Insert a new line.
*   **Ctrl+Up / Ctrl+Down**: Navigate command history.
*   **Ctrl+N**: New conversation.
*   **Ctrl+C**: Quit.

## Local Models (Ollama)

To use local models:
1.  Install [Ollama](https://ollama.com/).
2.  Pull a model: `ollama pull llama3`.
3.  In Michaelmas: Use `/model` command and select your Ollama model.

## MCP Support

Michaelmas supports the **Model Context Protocol (MCP)** to connect to external tools and resources.

1.  **Configure Servers:** Edit the `mcp_config.json` file in the project root.
    ```json
    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
        }
      }
    }
    ```
2.  **Restart Application:** Restart the TUI to load the new configuration.
3.  **Enable Tools:** Go to `/settings` to enable the discovered MCP tools.

## Maintenance

*   `make clean`: Removes the virtual environment, logs, and temporary files.

## License

MIT License. Copyright (c) 2025 Diego Amor (bl4ckb1rd).

## Roadmap

- [ ] **Database Integration**: Implement a persistent database (e.g., SQLite/PostgreSQL) for robust conversation storage and retrieval.
- [x] **MCP Support**: Add support for the Model Context Protocol (MCP) to standardize context management.
- [ ] **Expanded Toolset**: Integrate additional tools to enhance agent capabilities (e.g., calendar, email, file system).