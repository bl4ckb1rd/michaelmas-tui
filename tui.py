import os
import logging
from dotenv import load_dotenv
from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import Header, Footer, Input, TextArea, Label, OptionList

from langchain_core.messages import HumanMessage, AIMessage

from agent import run_agent
import storage

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
    ]

    class Submitted(Message):
        """Posted when the user presses Enter."""
        def __init__(self, value: str) -> None:
            super().__init__()
            self.value = value

    def action_submit_message(self) -> None:
        """Submit the text."""
        message = self.text.strip()
        if message:
            self.post_message(self.Submitted(message))
            self.clear()
    
    def action_insert_newline(self) -> None:
        """Insert a newline."""
        self.insert("\n")

class TuiApp(App):
    """A Textual app to chat with a LangGraph agent."""

    CSS_PATH = "tui.css"
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+n", "new_chat", "New Chat"),
    ]

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
        self.conversation_history = [] # List of BaseMessage objects

        self.load_conversations_into_sidebar()
        
        self.write_to_log("[bold green]Agent:[/] Hello! How can I assist you today?")
        self.write_to_log(f"[bold yellow]INFO:[/] Using model: {self.current_model}")
        
        self.update_status_bar(0.0)
        self.query_one(ChatInput).focus()

    def update_status_bar(self, last_cost: float):
        """Updates the status bar with current metrics."""
        status_text = f"Last: ${last_cost:.6f} | Session: ${self.session_total_cost:.6f} | Month: ${self.monthly_cost:.6f} | Tokens: {self.session_total_tokens}"
        self.query_one("#status", Label).update(status_text)

    def action_new_chat(self) -> None:
        """Starts a new conversation."""
        self.current_conversation_id = None
        self.conversation_history = []
        self.query_one("#log", TextArea).clear()
        self.write_to_log("[bold green]Agent:[/] Started a new conversation.")
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
        """Handle sidebar selection."""
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
            self.write_to_log(f"[bold red]Error:[/] Could not load conversation {conversation_id}")
            return

        self.current_conversation_id = data["id"]
        self.conversation_history = []
        self.query_one("#log", TextArea).clear()
        
        # Replay messages
        for msg in data.get("messages", []):
            if msg["type"] == "human":
                self.conversation_history.append(HumanMessage(content=msg["content"]))
                self.write_to_log(f"[bold blue]You:[/] {msg['content']}")
            elif msg["type"] == "ai":
                # Restore metadata
                additional_kwargs = {}
                if "cost" in msg:
                    additional_kwargs["cost"] = msg["cost"]
                if "usage" in msg:
                    additional_kwargs["usage"] = msg["usage"]
                
                self.conversation_history.append(AIMessage(content=msg["content"], additional_kwargs=additional_kwargs))
                self.write_to_log(f"[bold green]Agent:[/] {msg['content']}")
        
        self.write_to_log(f"[bold yellow]INFO:[/] Loaded conversation: {data.get('title')}")

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

    # --- Worker Methods ---

    def run_agent_worker(self) -> None:
        """Runs the agent in a background worker."""
        prompt = self.prompt_to_run
        model = self.current_model
        history = self.conversation_history.copy() # Pass copy of history

        logging.info(f"Running agent with model: {model}")
        try:
            result = run_agent(prompt, model_name=model, history=history)
            response = result.get("response", "No response")
            usage = result.get("usage", {})
            
            # Calculate costs
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
                    content=response, 
                    additional_kwargs={"cost": interaction_cost, "usage": usage}
                )
            )
            
            # Save persistence
            self.call_from_thread(self.save_current_chat)
            
            # Update status bar
            self.call_from_thread(self.update_status_bar, interaction_cost)
            
            self.call_from_thread(self.write_to_log, f"[bold green]Agent:[/] {response}")
            logging.info(f"Agent response: {response}")
            logging.info(f"Usage: {usage}, Cost: {interaction_cost}")
            
        except Exception as e:
            error_msg = f"[bold red]Error:[/] {e}"
            self.call_from_thread(self.write_to_log, error_msg)
            logging.error(f"Error running agent: {e}", exc_info=True)
        finally:
            self.call_from_thread(self.reset_input)

    def list_models_worker(self) -> None:
        """Provides a list of common Gemini models."""
        logging.info("Listing available models.")
        models_info = "[bold cyan]Available Gemini Models:[/]\n"
        models_info += "- `gemini-2.5-pro` (Most capable, multimodal)\n"
        models_info += "- `gemini-2.5-flash` (Fast, good for many tasks)\n"
        models_info += "- `gemini-2.5-flash-lite` (Even faster, smaller version)\n"
        models_info += "- `gemini-3-pro-preview` (Preview model, potentially highly capable)\n"
        models_info += "\n[bold yellow]Note:[/] These are the models you've indicated are available. For a comprehensive list and their specific capabilities, please refer to the official Google AI documentation."
        self.call_from_thread(self.write_to_log, models_info)

        self.call_from_thread(self.reset_input)

    def list_models_worker(self) -> None:
        """Provides a list of common Gemini models."""
        models_info = "[bold cyan]Available Gemini Models:[/]\n..."
        self.call_from_thread(self.write_to_log, models_info)
        self.call_from_thread(self.query_one(ChatInput).focus)
        self.call_from_thread(self.query_one(ChatInput).__setattr__, "disabled", False)


    # --- Event Handlers & UI Logic ---

    def write_to_log(self, text: str) -> None:
        """Helper method to write text to the log."""
        plain_text = Text.from_markup(text).plain
        log = self.query_one("#log", TextArea)
        log.insert(plain_text + "\n")
        log.scroll_end(animate=False)

    def action_toggle_dark(self) -> None:
        """An action to toggle dark mode."""
        self.dark = not self.dark
    
    async def on_chat_input_submitted(self, message: ChatInput.Submitted) -> None:
        """Called when the user submits a prompt or command."""
        text = message.value.strip()
        if not text:
            return

        input_widget = self.query_one(ChatInput)
        input_widget.disabled = True
        
        # Handle slash commands
        if text.startswith("/"):
            self.write_to_log(f"[bold yellow]CMD:[/] {text}")
            command, *args = text.split()
            
            if command == "/new":
                self.action_new_chat()
                input_widget.disabled = False
            elif command == "/list_models":
                self.run_worker(self.list_models_worker, exclusive=True, thread=True)
            elif command == "/set_model":
                if args:
                    self.current_model = args[0]
                    self.write_to_log(f"[bold yellow]INFO:[/] Model set to: {self.current_model}")
                    input_widget.disabled = False
                else:
                    self.write_to_log("[bold red]Error:[/] /set_model requires a model name.")
                    input_widget.disabled = False
            else:
                self.write_to_log(f"[bold red]Error:[/] Unknown command: {command}")
                input_widget.disabled = False
            
            input_widget.clear()
            input_widget.focus()

        # Handle regular prompts
        else:
            self.write_to_log(f"[bold blue]You:[/] {text}")
            self.write_to_log("[bold yellow]Agent is thinking...[/]")
            
            self.prompt_to_run = text
            self.run_worker(self.run_agent_worker, exclusive=True, thread=True)
            input_widget.clear()


if __name__ == "__main__":
    app = TuiApp()
    app.run()