import os
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from rich.text import Text
import ollama
from openai import OpenAI
from google.ai.generativelanguage import ModelServiceClient
from google.api_core.client_options import ClientOptions

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Grid
from textual.message import Message
from textual.widgets import Header, Footer, TextArea, Label, OptionList, Button, Input, SelectionList, Checkbox
from textual.widgets.option_list import Option
from textual.screen import ModalScreen

from langchain_core.messages import HumanMessage, AIMessage

from michaelmas.core.agent import run_agent, ALL_TOOLS_MAP
from michaelmas.core import storage

# Load environment variables from .env file
load_dotenv()

# Pricing per 1M tokens (approximate)
PRICING = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-2.5-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    "openai:gpt-4o": {"input": 5.00, "output": 15.00},
    "openai:gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "openai:gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

class ChatInput(TextArea):
    """Custom TextArea for chat input that handles Enter/Shift+Enter."""
    
    BINDINGS = [
        Binding("enter", "submit_message", "Send message", priority=True),
        Binding("shift+enter", "insert_newline", "New line", priority=True),
        Binding("ctrl+up", "history_prev", "Previous command", priority=True),
        Binding("ctrl+down", "history_next", "Next command", priority=True),
    ]

    class Submitted(Message):
        """Posted when the user presses Enter."""
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    class HistoryPrev(Message):
        """Posted when user presses Ctrl+Up."""
        pass

    class HistoryNext(Message):
        """Posted when user presses Ctrl+Down."""
        pass

    def action_submit_message(self) -> None:
        """Submit the text."""
        message = self.text.strip()
        if message:
            self.post_message(self.Submitted(message))
            self.clear()
    
    def action_insert_newline(self) -> None:
        """Insert a newline."""
        self.insert("\n")

    def action_history_prev(self) -> None:
        self.post_message(self.HistoryPrev())

    def action_history_next(self) -> None:
        self.post_message(self.HistoryNext())

class ModelSelectionScreen(ModalScreen[str]):
    """A modal screen to select a model."""

    def __init__(self, models: list[str]):
        super().__init__()
        self.models = models

    def compose(self) -> ComposeResult:
        yield Grid(
            Label("Select an AI Model", id="model-title"),
            OptionList(*self.models, id="model-list"),
            id="model-dialog",
        )

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        selected_model = str(event.option.prompt)
        self.dismiss(selected_model)

    def key_escape(self) -> None:
        self.dismiss(None)

class SettingsScreen(ModalScreen[dict]):
    """A modal screen to configure settings."""

    def __init__(self, current_temp: float, current_max_tokens: int | None, all_tools: list[str] = [], enabled_tools: list[str] = [], tools_enabled_master: bool = True):
        super().__init__()
        self.current_temp = current_temp
        self.current_max_tokens = current_max_tokens
        self.all_tools = all_tools
        self.enabled_tools = enabled_tools
        self.tools_enabled_master = tools_enabled_master

    def compose(self) -> ComposeResult:
        max_tokens_val = str(self.current_max_tokens) if self.current_max_tokens is not None else "" 
        
        # Prepare tool selections
        tool_selections = []
        for tool in self.all_tools:
            tool_selections.append((tool, tool, tool in self.enabled_tools))

        yield Grid(
            Label("LLM Settings", id="settings-title"),
            
            Label("Temperature (0.0 - 1.0):"),
            Input(str(self.current_temp), id="temp-input", type="number", placeholder="0.7"),
            
            Label("Max Output Tokens (Optional):"),
            Input(max_tokens_val, id="tokens-input", type="number", placeholder="e.g., 2048"),
            
            Label("Tool Configuration:", id="tools-header"),
            Checkbox("Enable Tool Use", value=self.tools_enabled_master, id="master-tool-check"),
            
            Label("Select Tools:", id="tools-label"),
            SelectionList(*tool_selections, id="tools-list", disabled=not self.tools_enabled_master),

            Button("Save", variant="primary", id="save-btn"),
            id="settings-dialog",
        )

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "master-tool-check":
            self.query_one("#tools-list", SelectionList).disabled = not event.value

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            try:
                temp_val = float(self.query_one("#temp-input", Input).value)
                if not (0.0 <= temp_val <= 1.0):
                    self.app.notify("Temperature must be between 0.0 and 1.0", severity="error")
                    return

                tokens_val_str = self.query_one("#tokens-input", Input).value
                tokens_val = int(tokens_val_str) if tokens_val_str else None
                
                selected_tools = self.query_one("#tools-list", SelectionList).selected
                master_enabled = self.query_one("#master-tool-check", Checkbox).value

                self.dismiss({
                    "temperature": temp_val, 
                    "max_tokens": tokens_val,
                    "enabled_tools": selected_tools,
                    "tools_enabled_master": master_enabled
                })
            except ValueError:
                self.app.notify("Invalid input values", severity="error")
    
    def key_escape(self) -> None:
        """Handle Escape key to cancel."""
        self.dismiss(None)

class TuiApp(App):
# ... (rest of TuiApp) ...    """A Textual app to chat with a LangGraph agent."""

    TITLE = "Michaelmas"
    CSS_PATH = "styles.css"
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+n", "new_chat", "New Chat"),
    ]

    class ModelsListed(Message):
        """Posted when list_models_worker returns available models."""
        def __init__(self, models: list[str], info_text: str) -> None:
            super().__init__()
            self.models = models
            self.info_text = info_text

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        
        # Main Body (Horizontal Split: Sidebar + Chat)
        with Container(id="app-grid"):
            yield OptionList(id="sidebar")
            with Vertical(id="main-content"):
                yield TextArea(id="log", read_only=True, language="log")
                yield ChatInput(id="input", show_line_numbers=False)
        
        # Global Status Bar
        yield Label("Initializing...", id="status")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        logging.info("TuiApp mounted.")
        self.dark = False
        self.current_model = "gemini-2.5-flash"
        self.llm_temperature = 0.7 # Default temperature
        self.llm_max_tokens = None # Default max tokens (None means unlimited/model default) 
        
        self.session_total_cost = 0.0
        self.session_total_tokens = 0
        self.monthly_cost = storage.calculate_monthly_cost()
        
        # Conversation State
        self.current_conversation_id = None
        self.conversation_history = [] 
        self.conversation_cost = 0.0
        self.conversation_tokens = 0
        
        # Command History
        self.command_history = []
        self.history_index = -1 

        # Model State
        self.available_models: list[str] = []
        self.enabled_tools: list[str] = list(ALL_TOOLS_MAP.keys())
        self.tools_enabled_master: bool = True
        
        # Model Cache
        self.cached_gemini_models: list[str] | None = None
        self.cached_openai_models: list[str] | None = None

        self.load_conversations_into_sidebar()
        
        self.write_to_log("[bold green]Agent:[/ ] Hello! How can I assist you today?")
        self.write_to_log(f"[bold yellow]INFO:[/ ] Using model: {self.current_model}")
        
        self.update_status_bar(0.0) # Initialize with 0 for last cost
        self.query_one(ChatInput).focus()
        
        # Sidebar auto-refresh
        self.set_interval(10, self.load_conversations_into_sidebar)

    # --- Helper Methods ---

    def update_status_bar(self, last_cost: float):
        """Updates the status bar with current metrics."""
        model_display = f"[bold]{self.current_model}[/bold]"
        
        tokens_display = f"MaxTok={self.llm_max_tokens}" if self.llm_max_tokens else "MaxTok=Auto"
        
        status_text = (
            f"Model: {model_display} (T={self.llm_temperature}, {tokens_display}) | "
            f"Last: [green]${last_cost:.6f}[/] | "
            f"Conv: [blue]${self.conversation_cost:.6f}[/] ({self.conversation_tokens} tok) | "
            f"Month: [red]${self.monthly_cost:.6f}[/]"
        )
        self.query_one("#status", Label).update(status_text)

    def action_new_chat(self) -> None:
        """Starts a new conversation."""
        self.current_conversation_id = None
        self.conversation_history = []
        self.conversation_cost = 0.0
        self.conversation_tokens = 0
        self.query_one("#log", TextArea).clear()
        self.write_to_log("[bold green]Agent:[/ ] Started a new conversation.")
        self.load_conversations_into_sidebar() # Refresh list (to unselect)
        self.update_status_bar(0.0) # Reset status bar

    def load_conversations_into_sidebar(self):
        """Loads conversations from storage into the sidebar, grouped by date."""
        sidebar = self.query_one("#sidebar", OptionList)
        
        # Store current selection index to restore if possible?
        
        sidebar.clear_options()
        
        all_conversations = storage.list_conversations()
        if not all_conversations:
            return

        # buckets
        groups = {
            "Today": [],
            "Yesterday": [],
            "Previous 7 Days": [],
            "Previous 30 Days": [],
            "Older": []
        }
        
        now = datetime.now()
        today = now.date()
        yesterday = today - timedelta(days=1)
        
        for conv in all_conversations:
            try:
                updated_at = datetime.fromisoformat(conv["updated_at"]).date()
            except (ValueError, TypeError):
                groups["Older"].append(conv)
                continue

            if updated_at == today:
                groups["Today"].append(conv)
            elif updated_at == yesterday:
                groups["Yesterday"].append(conv)
            elif updated_at > today - timedelta(days=7):
                groups["Previous 7 Days"].append(conv)
            elif updated_at > today - timedelta(days=30):
                groups["Previous 30 Days"].append(conv)
            else:
                groups["Older"].append(conv)

        # Flatten for display and mapping
        self.conversations_map = {} 
        current_index = 0
        
        for group_name, convs in groups.items():
            if not convs:
                continue
            
            # Add Header
            sidebar.add_option(Option(f"── {group_name} ──", disabled=True))
            self.conversations_map[current_index] = None # Header is not clickable
            current_index += 1
            
            for conv in convs:
                title = conv.get("title", "Untitled")
                title = title.split("\n")[0][:30] 
                
                sidebar.add_option(Option(title, id=conv["id"]))
                self.conversations_map[current_index] = conv
                current_index += 1

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle sidebar selection (Main conversation list)."""
        if event.option_list.id == "sidebar":
            if event.option_index is None:
                return
            
            selected_item = self.conversations_map.get(event.option_index)
            if selected_item: 
                self.load_chat(selected_item["id"])

    def load_chat(self, conversation_id: str):
        """Loads a specific conversation."""
        data = storage.load_conversation(conversation_id)
        if not data:
            self.write_to_log(f"[bold red]Error:[/ ] Could not load conversation {conversation_id}")
            return

        self.current_conversation_id = data["id"]
        self.conversation_history = []
        self.conversation_cost = 0.0 
        self.conversation_tokens = 0 
        self.query_one("#log", TextArea).clear()
        
        for msg in data.get("messages", []):
            if msg["type"] == "human":
                self.conversation_history.append(HumanMessage(content=msg["content"]))
                self.write_to_log(f"[bold blue]You:[/ ] {msg['content']}")
            elif msg["type"] == "ai":
                additional_kwargs = {}
                cost_per_msg = msg.get("cost", 0.0)
                usage_per_msg = msg.get("usage", {})
                
                additional_kwargs["cost"] = cost_per_msg
                additional_kwargs["usage"] = usage_per_msg
                
                self.conversation_history.append(AIMessage(content=msg["content"], additional_kwargs=additional_kwargs))
                self.write_to_log(f"[bold green]Agent:[/ ] {msg['content']}")
                
                self.conversation_cost += cost_per_msg
                self.conversation_tokens += usage_per_msg.get("total_tokens", 0)
        
        self.update_status_bar(0.0)
        self.write_to_log(f"[bold yellow]INFO:[/ ] Loaded conversation: {data.get('title')}")

    def save_current_chat(self):
        """Saves the current conversation to storage."""
        messages_data = []
        for msg in self.conversation_history:
            if isinstance(msg, HumanMessage):
                messages_data.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                msg_data = {"type": "ai", "content": msg.content}
                if "cost" in msg.additional_kwargs:
                    msg_data["cost"] = msg.additional_kwargs["cost"]
                if "usage" in msg.additional_kwargs:
                    msg_data["usage"] = msg.additional_kwargs["usage"]
                messages_data.append(msg_data)
        
        self.current_conversation_id = storage.save_conversation(
            self.current_conversation_id, 
            messages_data
        )

    def reset_input(self) -> None:
        """Re-enables, clears, and focuses the input widget."""
        input_widget = self.query_one(ChatInput)
        input_widget.disabled = False
        input_widget.clear()
        input_widget.focus()

    def set_available_models(self, models: list[str]) -> None:
        """Sets the available models list on the main thread."""
        self.available_models = models

    def stream_text_to_log(self, text: str) -> None:
        """Appends text to the log without a newline (for streaming)."""
        plain_text = Text.from_markup(text).plain
        log = self.query_one("#log", TextArea)
        log.insert(plain_text)
        log.scroll_end(animate=False)

    def write_to_log(self, text: str) -> None:
        """Helper method to write text to the log and scroll to the end."""
        plain_text = Text.from_markup(text).plain
        log = self.query_one("#log", TextArea)
        log.insert(plain_text + "\n")
        log.scroll_end(animate=False)

    # --- Worker Methods ---

    async def run_agent_worker(self) -> None:
        """Runs the agent in a background worker with streaming."""
        prompt = self.prompt_to_run
        model = self.current_model
        temperature = self.llm_temperature
        max_tokens = self.llm_max_tokens
        
        # Determine enabled tools based on master switch
        enabled_tools = self.enabled_tools if self.tools_enabled_master else []
        
        history = self.conversation_history.copy() # Pass copy of history

        logging.info(f"Running agent with model: {model}, temp: {temperature}, max_tok: {max_tokens}, tools: {len(enabled_tools)}")
        
        self.call_from_thread(self.write_to_log, "[bold green]Agent:[/ ]") 
        
        full_response = ""
        usage = {}
        interaction_cost = 0.0

        try:
            async for chunk in run_agent(prompt, model_name=model, temperature=temperature, max_tokens=max_tokens, history=history, enabled_tools=enabled_tools):
                if chunk["type"] == "token":
                    text = chunk["content"]
                    full_response += text
                    self.call_from_thread(self.stream_text_to_log, text)
                
                elif chunk["type"] == "usage":
                    usage = chunk["usage"]
            
            self.call_from_thread(self.stream_text_to_log, "\n")

            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            pricing = PRICING.get(model, {"input": 0, "output": 0})
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            interaction_cost = input_cost + output_cost
            
            self.session_total_cost += interaction_cost
            self.session_total_tokens += total_tokens
            self.monthly_cost += interaction_cost
            self.conversation_cost += interaction_cost
            self.conversation_tokens += total_tokens
            
            self.conversation_history.append(HumanMessage(content=prompt))
            self.conversation_history.append(
                AIMessage(
                    content=full_response, 
                    additional_kwargs={"cost": interaction_cost, "usage": usage}
                )
            )
            
            self.call_from_thread(self.save_current_chat)
            self.call_from_thread(self.update_status_bar, interaction_cost)
            
            logging.info(f"Agent response (streamed): {full_response}")
            logging.info(f"Usage: {usage}, Cost: {interaction_cost}")
            
        except Exception as e:
            error_msg = f"\n[bold red]Error:[/ ] {e}"
            self.call_from_thread(self.write_to_log, error_msg)
            logging.error(f"Error running agent: {e}", exc_info=True)
        finally:
            self.call_from_thread(self.reset_input)

    def select_model_worker(self) -> None:
        """Fetches models and opens the selection modal."""
        logging.info("Fetching models for selection.")
        
        # 1. Gemini Models
        gemini_models = []
        if self.cached_gemini_models is not None:
            gemini_models = self.cached_gemini_models
        elif "GOOGLE_API_KEY" in os.environ:
            try:
                client = ModelServiceClient(
                    client_options=ClientOptions(api_key=os.environ["GOOGLE_API_KEY"])
                )
                for m in client.list_models():
                    if "generateContent" in m.supported_generation_methods:
                        name = m.name.replace("models/", "")
                        gemini_models.append(name)
                self.cached_gemini_models = gemini_models
            except Exception as e:
                logging.warning(f"Failed to list Gemini models: {e}")
        
        if not gemini_models: # Fallback
            gemini_models = [
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite",
                "gemini-3-pro-preview",
            ]

        # 2. OpenAI Models
        openai_models = []
        if self.cached_openai_models is not None:
            openai_models = self.cached_openai_models
        elif "OPENAI_API_KEY" in os.environ:
            try:
                client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
                for m in client.models.list():
                    if m.id.startswith("gpt"):
                        openai_models.append(f"openai:{m.id}")
                self.cached_openai_models = openai_models
            except Exception as e:
                logging.warning(f"Failed to list OpenAI models: {e}")

        # 3. Ollama Models
        ollama_models = []
        try:
            ollama_response = ollama.list()
            if 'models' in ollama_response:
                for m in ollama_response['models']:
                    name = m['model']
                    ollama_models.append(f"ollama:{name}")
        except Exception as e:
            logging.warning(f"Failed to list Ollama models: {e}")
        
        all_models = sorted(list(set(gemini_models))) + sorted(openai_models) + sorted(ollama_models)
        
        self.call_from_thread(self.app.push_screen, ModelSelectionScreen(all_models), self.on_model_selected)

    def select_settings_worker(self) -> None:
        """Opens the settings modal."""
        all_tools = list(ALL_TOOLS_MAP.keys())
        self.call_from_thread(self.app.push_screen, SettingsScreen(self.llm_temperature, self.llm_max_tokens, all_tools, self.enabled_tools, self.tools_enabled_master), self.on_settings_changed)

    def list_models_worker(self) -> None:
        """Provides a list of available models (Gemini + Ollama)."""
        logging.info("Listing available models.")
        models_info = "[bold cyan]Available Gemini Models (for your API key):[/]\n"
        gemini_models = [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3-pro-preview",
        ]
        
        ollama_models = []
        try:
            ollama_response = ollama.list()
            if 'models' in ollama_response:
                for m in ollama_response['models']:
                    name = m['model']
                    ollama_models.append(f"ollama:{name}")
            else:
                 models_info += "[dim]No models found in Ollama response.[/]\n"
        except Exception as e:
            logging.warning(f"Failed to list Ollama models: {e}")
            models_info += f"[dim]Could not connect to Ollama ({e}). Make sure it is running.[/]\n"
        
        all_models = gemini_models + ollama_models
        
        models_info += "\n[bold yellow]Note:[/ ] Use `/set_model <name>` to switch."
        
        self.call_from_thread(self.write_to_log, models_info)
        self.call_from_thread(self.set_available_models, all_models)
        self.call_from_thread(self.reset_input)


    # --- Event Handlers (Callbacks) ---

    def on_models_listed(self, message: ModelsListed) -> None:
        """Handles the ModelsListed message to store and display available models."""
        self.available_models = message.models
        self.write_to_log(message.info_text) 
        self.reset_input()

    def on_model_selected(self, model_name: str) -> None:
        """Callback for when a model is selected from the modal."""
        if model_name:
            self.current_model = model_name
            self.write_to_log(f"[bold yellow]INFO:[/ ] Model set to: {self.current_model}")
            logging.info(f"Model switched to: {self.current_model}")
            self.update_status_bar(0.0) 
        
        self.reset_input()

    def on_settings_changed(self, settings: dict) -> None:
        """Callback for settings change."""
        if settings:
            self.llm_temperature = settings.get("temperature", 0.0)
            self.llm_max_tokens = settings.get("max_tokens")
            
            if "enabled_tools" in settings:
                self.enabled_tools = settings["enabled_tools"]
            
            if "tools_enabled_master" in settings:
                self.tools_enabled_master = settings["tools_enabled_master"]

            log_msg = f"[bold yellow]INFO:[/ ] Settings updated: T={self.llm_temperature}"
            if self.llm_max_tokens:
                log_msg += f", MaxTokens={self.llm_max_tokens}"
            
            if not self.tools_enabled_master:
                log_msg += ", Tools=DISABLED"
            elif self.enabled_tools is not None:
                log_msg += f", Tools={len(self.enabled_tools)}"
            
            self.write_to_log(log_msg)
            
            self.update_status_bar(0.0)
        
        self.reset_input()

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark
    
    async def on_chat_input_submitted(self, message: ChatInput.Submitted) -> None:
        """Called when the user submits a prompt or command."""
        text = message.value.strip()
        if not text:
            return

        # Save to history
        self.command_history.append(text)
        self.history_index = -1

        input_widget = self.query_one(ChatInput)
        input_widget.disabled = True
        
        # Handle slash commands
        if text.startswith("/"):
            logging.info(f"User command: {text}")
            self.write_to_log(f"[bold yellow]CMD:[/ ] {text}")
            command, *args = text.split()
            
            if command == "/model":
                self.run_worker(self.select_model_worker, exclusive=True, thread=True)
            elif command == "/settings":
                self.run_worker(self.select_settings_worker, exclusive=True, thread=True)
            elif command == "/new":
                self.action_new_chat()
                self.reset_input()
            elif command == "/list_models": # Kept for backward compatibility
                self.run_worker(self.list_models_worker, exclusive=True, thread=True)
            elif command == "/help":
                help_text = """[bold cyan]Available Commands:[/ ]
- [bold]/help[/ ]: Show this help message.
- [bold]/new[/ ]: Start a new conversation.
- [bold]/model[/ ]: Open the model selector to switch AI models.
- [bold]/settings[/ ]: Configure LLM settings (e.g., temperature, tools).
"""
                self.write_to_log(help_text)
                self.reset_input()
            else:
                self.write_to_log(f"[bold red]Error:[/ ] Unknown command: {command}")
                logging.warning(f"Unknown command received: {command}")
                self.reset_input()
            
        # Handle regular prompts
        else:
            logging.info(f"User prompt: {text}")
            self.write_to_log(f"[bold blue]You:[/ ] {text}")
            self.write_to_log("[bold yellow]Agent is thinking...[/]")
            
            self.prompt_to_run = text
            self.run_worker(self.run_agent_worker, exclusive=True, thread=True)


if __name__ == "__main__":
    app = TuiApp()
    app.run()
