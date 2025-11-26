import os
import logging
from dotenv import load_dotenv
from rich.text import Text
import ollama

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Grid
from textual.message import Message
from textual.widgets import Header, Footer, TextArea, Label, OptionList
from textual.screen import ModalScreen

from langchain_core.messages import HumanMessage, AIMessage

from michaelmas.core.agent import run_agent
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
        # OptionList returns the string label of the selected option
        selected_model = str(event.option.prompt)
        self.dismiss(selected_model)

class TuiApp(App):
    """A Textual app to chat with a LangGraph agent."""

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
        with Container(id="app-grid"):
            yield OptionList(id="sidebar")
            with Vertical(id="main-content"):
                yield TextArea(id="log", read_only=True, language="log")
                yield ChatInput(id="input", show_line_numbers=False)
                yield Label("Cost: $0.0000 | Session: $0.0000 | Month: $0.0000 | Tokens: 0", id="status")
        yield Footer()
    
    def on_mount(self) -> None:
        """Called when the app is mounted."""
        logging.info("TuiApp mounted.")
        self.dark = False
        self.current_model = "gemini-2.5-flash"
        self.session_total_cost = 0.0
        self.session_total_tokens = 0
        self.monthly_cost = storage.calculate_monthly_cost()
        
        # Conversation State
        self.current_conversation_id = None
        self.conversation_history = [] 
        
        # Command History
        self.command_history = []
        self.history_index = -1 

        # Model State
        self.available_models: list[str] = []

        self.load_conversations_into_sidebar()
        
        self.write_to_log("[bold green]Agent:[/ ] Hello! How can I assist you today?")
        self.write_to_log(f"[bold yellow]INFO:[/ ] Using model: {self.current_model}")
        
        self.update_status_bar(0.0)
        self.query_one(ChatInput).focus()

    def on_chat_input_history_prev(self) -> None:
        """Handle Ctrl+Up."""
        if not self.command_history:
            return
        
        # If we are currently at "new line" (index -1), start from the end
        if self.history_index == -1:
            self.history_index = len(self.command_history) - 1
        elif self.history_index > 0:
            self.history_index -= 1
        
        self._load_history_item()

    def on_chat_input_history_next(self) -> None:
        """Handle Ctrl+Down."""
        if self.history_index == -1:
            return

        if self.history_index < len(self.command_history) - 1:
            self.history_index += 1
            self._load_history_item()
        else:
            # We are at the end, pushing down goes back to empty line
            self.history_index = -1
            self.query_one(ChatInput).clear()

    def _load_history_item(self):
        """Helper to load the command at current history_index into input."""
        if 0 <= self.history_index < len(self.command_history):
            text = self.command_history[self.history_index]
            input_widget = self.query_one(ChatInput)
            input_widget.clear()
            input_widget.insert(text)

    def update_status_bar(self, last_cost: float):
        """Updates the status bar with current metrics."""
        status_text = f"Last: ${last_cost:.6f} | Session: ${self.session_total_cost:.6f} | Month: ${self.monthly_cost:.6f} | Tokens: {self.session_total_tokens}"
        self.query_one("#status", Label).update(status_text)

    def action_new_chat(self) -> None:
        """Starts a new conversation."""
        self.current_conversation_id = None
        self.conversation_history = []
        self.query_one("#log", TextArea).clear()
        self.write_to_log("[bold green]Agent:[/ ] Started a new conversation.")
        self.load_conversations_into_sidebar() # Refresh list (to unselect)

    def load_conversations_into_sidebar(self):
        """Loads conversations from storage into the sidebar."""
        sidebar = self.query_one("#sidebar", OptionList)
        sidebar.clear_options()
        conversations = storage.list_conversations()
        for conv in conversations:
            sidebar.add_option(f"{conv['title']} ({conv['id'][:4]})") 
        self.conversations_map = conversations 

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle sidebar selection (Main conversation list)."""
        # Only handle sidebar selections here (check ID)
        if event.option_list.id == "sidebar":
            if event.option_index is None:
                return
            
            try:
                selected_conv = self.conversations_map[event.option_index]
                self.load_chat(selected_conv["id"])
            except IndexError:
                pass

    def load_chat(self, conversation_id: str):
        """Loads a specific conversation."""
        data = storage.load_conversation(conversation_id)
        if not data:
            self.write_to_log(f"[bold red]Error:[/ ] Could not load conversation {conversation_id}")
            return

        self.current_conversation_id = data["id"]
        self.conversation_history = []
        self.query_one("#log", TextArea).clear()
        
        # Replay messages
        for msg in data.get("messages", []):
            if msg["type"] == "human":
                self.conversation_history.append(HumanMessage(content=msg["content"]))
                self.write_to_log(f"[bold blue]You:[/ ] {msg['content']}")
            elif msg["type"] == "ai":
                # Restore metadata
                additional_kwargs = {}
                if "cost" in msg:
                    additional_kwargs["cost"] = msg["cost"]
                if "usage" in msg:
                    additional_kwargs["usage"] = msg["usage"]
                
                self.conversation_history.append(AIMessage(content=msg["content"], additional_kwargs=additional_kwargs))
                self.write_to_log(f"[bold green]Agent:[/ ] {msg['content']}")
        
        self.write_to_log(f"[bold yellow]INFO:[/ ] Loaded conversation: {data.get('title')}")

    def save_current_chat(self):
        """Saves the current conversation to storage."""
        # Convert BaseMessage objects to dicts
        messages_data = []
        for msg in self.conversation_history:
            if isinstance(msg, HumanMessage):
                messages_data.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                msg_data = {"type": "ai", "content": msg.content}
                # Save metadata if present
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
        # Strip markup since we are in TextArea mode
        plain_text = Text.from_markup(text).plain
        log = self.query_one("#log", TextArea)
        log.insert(plain_text)
        log.scroll_end(animate=False)

    def write_to_log(self, text: str) -> None:
        """Helper method to write text to the log and scroll to the end."""
        # Strip markup tags so they don't appear as raw text in the TextArea
        plain_text = Text.from_markup(text).plain
        log = self.query_one("#log", TextArea)
        log.insert(plain_text + "\n")
        log.scroll_end(animate=False)

    # --- Worker Methods ---

    async def run_agent_worker(self) -> None:
        """Runs the agent in a background worker with streaming."""
        prompt = self.prompt_to_run
        model = self.current_model
        history = self.conversation_history.copy() # Pass copy of history

        logging.info(f"Running agent with model: {model}")
        
        # Start the log entry for the agent
        self.call_from_thread(self.write_to_log, "[bold green]Agent:[/]") 
        
        full_response = ""
        usage = {}
        interaction_cost = 0.0

        try:
            # run_agent is now an async generator
            async for chunk in run_agent(prompt, model_name=model, history=history):
                if chunk["type"] == "token":
                    text = chunk["content"]
                    full_response += text
                    self.call_from_thread(self.stream_text_to_log, text)
                
                elif chunk["type"] == "usage":
                    usage = chunk["usage"]
            
            # Add a newline after streaming finishes
            self.call_from_thread(self.stream_text_to_log, "\n")

            # Calculate costs (post-stream)
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            pricing = PRICING.get(model, {"input": 0, "output": 0})
            input_cost = (input_tokens / 1_000_000) * pricing["input"]
            output_cost = (output_tokens / 1_000_000) * pricing["output"]
            interaction_cost = input_cost + output_cost
            
            # Update totals
            self.session_total_cost += interaction_cost
            self.session_total_tokens += total_tokens
            self.monthly_cost += interaction_cost
            
            # Update history with metadata
            self.conversation_history.append(HumanMessage(content=prompt))
            self.conversation_history.append(
                AIMessage(
                    content=full_response, 
                    additional_kwargs={"cost": interaction_cost, "usage": usage}
                )
            )
            
            # Save persistence
            self.call_from_thread(self.save_current_chat)
            
            # Update status bar
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
        except Exception as e:
            logging.warning(f"Failed to list Ollama models: {e}")
        
        all_models = gemini_models + ollama_models
        
        # Push the screen from the worker, but we must ensure we're calling App methods safely.
        # push_screen is thread safe in Textual.
        self.call_from_thread(self.app.push_screen, ModelSelectionScreen(all_models), self.on_model_selected)

    def on_model_selected(self, model_name: str) -> None:
        """Callback for when a model is selected from the modal."""
        if model_name:
            self.current_model = model_name
            self.write_to_log(f"[bold yellow]INFO:[/ ] Model set to: {self.current_model}")
            logging.info(f"Model switched to: {self.current_model}")
        
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
            elif command == "/help":
                help_text = """[bold cyan]Available Commands:[/ ]
- [bold]/help[/ ]: Show this help message.
- [bold]/new[/ ]: Start a new conversation.
- [bold]/model[/ ]: Open the model selector to switch AI models.
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