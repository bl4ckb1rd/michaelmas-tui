import os
import logging
from typing import TypedDict, Annotated, Sequence, Literal
import operator
from datetime import datetime
from dotenv import load_dotenv
import ollama

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# --- Import Tools ---
from michaelmas.tools.sheets import (
    create_spreadsheet,
    get_sheet_data,
    update_sheet_data,
    append_to_sheet,
)
from michaelmas.tools.search import web_search
from michaelmas.tools.shell import run_shell_command

# Load environment variables from .env file
load_dotenv()

# Ensure the Google API key is set (only strictly needed if using Gemini)
if "GOOGLE_API_KEY" not in os.environ:
    logging.warning("GOOGLE_API_KEY not found. Gemini models will not work.")

# --- System Prompt ---
DEFAULT_SYSTEM_PROMPT = """
You are a helpful and versatile AI assistant. You can use tools to perform tasks.
You are equipped to handle Google Sheets operations, search the web for information, and execute shell commands on the local system.
Your goal is to assist the user by understanding their requests and utilizing the available tools efficiently.
Always respond in a concise and helpful manner.
"""

# --- 1. Define Tools ---
sheets_tools = [
    create_spreadsheet,
    get_sheet_data,
    update_sheet_data,
    append_to_sheet,
]
general_tools = [web_search, run_shell_command]

# --- 2. Supervisor Agent Logic ---

class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: Literal["agent_sheets", "agent_general", "end"]

class RouteQuery(BaseModel):
    """Route a user query to the appropriate agent."""
    destination: Literal["sheets", "general"] = Field(
        ...,
        description="Given a user question, route it to the appropriate agent: 'sheets' for Google Sheets tasks or 'general' for all other questions (including web search, news, and chitchat).",
    )

def create_supervisor(llm):
    structured_llm_router = llm.with_structured_output(RouteQuery)

    def supervisor_node(state: SupervisorState):
        logging.debug("SUPERVISOR: Checking for routing...")
        last_message = state["messages"][-1]
        
        if isinstance(last_message, ToolMessage):
             logging.debug("SUPERVISOR: Last message was tool output, routing to END.")
             return {"next": "end"}
        
        route = structured_llm_router.invoke(state["messages"])
        logging.info(f"SUPERVISOR: Route found: {route.destination}")
        if route.destination == "sheets":
            return {"next": "agent_sheets"}
        else:
            return {"next": "agent_general"}

    return supervisor_node

# --- 3. Agent Nodes ---

# Helper to check Ollama tool support dynamically
def check_ollama_tool_support(model_name: str) -> bool:
    """
    Checks if an Ollama model supports tools by inspecting its template
    OR by checking against a known list of capable models.
    """
    # 1. Check template for {{ .Tools }}
    try:
        model_info = ollama.show(model_name)
        template = model_info.get('template', '')
        if "{{ .Tools }}" in template:
            return True
    except Exception as e:
        logging.warning(f"Failed to check template for {model_name}: {e}")

    # 2. Fallback: Check known capable models list
    # Some versions of models/ollama might not expose .Tools in template but still support it
    known_capable = ["llama3.1", "mistral", "mixtral", "qwen2.5", "command-r", "hermes"]
    model_name_lower = model_name.lower()
    if any(k in model_name_lower for k in known_capable):
        return True
        
    return False

def create_agent_node(llm, tools):
    def agent_node(state: SupervisorState):
        agent_name = tools[0].name.split('_')[0] if tools else "General"
        logging.info(f"AGENT ({agent_name}): Running...")
        
        # Conditional tool binding
        if isinstance(llm, ChatOllama):
            model_id = llm.model
            if check_ollama_tool_support(model_id):
                logging.info(f"AGENT ({agent_name}): Binding tools for capable Ollama model: {model_id}")
                model = llm.bind_tools(tools)
            else:
                logging.info(f"AGENT ({agent_name}): Skipping tool binding for Ollama model: {model_id} (No {{ .Tools }} in template)")
                model = llm
        else:
            # Always bind for Gemini/OpenAI
            model = llm.bind_tools(tools)
            
        response = model.invoke(state["messages"])
        return {"messages": [response]}
    return agent_node

def create_tool_node(tools):
    def tool_node(state: SupervisorState):
        logging.info("TOOL NODE: Executing tools...")
        last_message = state["messages"][-1]
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_to_call = {t.name: t for t in tools}[tool_name]
            try:
                logging.debug(f"TOOL NODE: Calling {tool_name} with args: {tool_call['args']}")
                output = tool_to_call.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(content=str(output), tool_call_id=tool_call["id"])
                )
            except Exception as e:
                logging.error(f"TOOL NODE: Error calling {tool_name}: {e}")
                tool_messages.append(
                    ToolMessage(content=f"Error: {e}", tool_call_id=tool_call["id"])
                )
        return {"messages": tool_messages}
    return tool_node

# --- 4. Graph Creation & Caching ---

# A cache for compiled agent graphs to avoid recompiling on every call
_agent_graph_cache = {}

def create_agent_graph(model_name: str = "gemini-2.5-flash", temperature: float = 0.7, max_tokens: int | None = None):
    """
    Creates and compiles the agent graph for a given model name, temperature, and max tokens.
    Caches the compiled graph to avoid redundant work.
    """
    cache_key = (model_name, temperature, max_tokens)
    if cache_key in _agent_graph_cache:
        return _agent_graph_cache[cache_key]

    # Instantiate the appropriate LLM
    if model_name.startswith("ollama:"):
        ollama_model = model_name.split(":", 1)[1]
        logging.info(f"Using ChatOllama with model: {ollama_model}, temperature: {temperature}, max_tokens: {max_tokens}")
        llm = ChatOllama(model=ollama_model, temperature=temperature, num_predict=max_tokens)
    elif model_name.startswith("openai:"):
        openai_model = model_name.split(":", 1)[1]
        logging.info(f"Using ChatOpenAI with model: {openai_model}, temperature: {temperature}, max_tokens: {max_tokens}")
        llm = ChatOpenAI(model=openai_model, temperature=temperature, max_tokens=max_tokens)
    else:
        logging.info(f"Using ChatGoogleGenerativeAI with model: {model_name}, temperature: {temperature}, max_tokens: {max_tokens}")
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_output_tokens=max_tokens)
    
    supervisor_node = create_supervisor(llm)
    
    # Sheets Sub-graph elements
    sheets_agent_node = create_agent_node(llm, sheets_tools)
    sheets_tool_node = create_tool_node(sheets_tools)
    
    # General Sub-graph elements (now includes search)
    general_agent_node = create_agent_node(llm, general_tools)
    general_tool_node = create_tool_node(general_tools)
    
    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", supervisor_node)
    
    workflow.add_node("agent_sheets", sheets_agent_node)
    workflow.add_node("tool_sheets", sheets_tool_node)
    
    workflow.add_node("agent_general", general_agent_node)
    workflow.add_node("tool_general", general_tool_node)
    
    workflow.set_entry_point("supervisor")
    
    # Supervisor Routing
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "agent_sheets": "agent_sheets", 
            "agent_general": "agent_general",
            "end": END
        },
    )
    
    # Helper for conditional edges inside sub-agents
    def should_continue(state: SupervisorState):
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "continue"
        return "end"

    # Sheets Loop Logic
    workflow.add_conditional_edges(
        "agent_sheets",
        should_continue,
        {
            "continue": "tool_sheets",
            "end": END
        }
    )
    workflow.add_edge("tool_sheets", "agent_sheets") # Return to agent to digest tool output
    
    # General Loop Logic
    workflow.add_conditional_edges(
        "agent_general",
        should_continue,
        {
            "continue": "tool_general",
            "end": END
        }
    )
    workflow.add_edge("tool_general", "agent_general") # Return to agent to digest tool output

    graph = workflow.compile()
    _agent_graph_cache[cache_key] = graph
    return graph

# --- 5. Main execution function ---

async def run_agent(prompt: str, model_name: str = "gemini-2.5-flash", temperature: float = 0.7, max_tokens: int | None = None, history: list[BaseMessage] = []):
    """
    Runs the agent with a given prompt, model name, temperature, and max tokens.
    Yields dictionaries with response tokens and usage metadata.
    """
    # Prepend current date to ensure agent has up-to-date context
    today_date_str = datetime.now().strftime("Today's date is %A, %B %d, %Y.")
    
    # Construct initial messages with system context
    initial_messages = []

    # If the history is empty or only contains a date message, add the default system prompt
    has_user_ai_messages = any(isinstance(msg, (HumanMessage, AIMessage)) for msg in history)
    
    if not has_user_ai_messages:
        initial_messages.append(SystemMessage(content=DEFAULT_SYSTEM_PROMPT))
    
    initial_messages.append(SystemMessage(content=today_date_str))

    # Combine initial context, history, and current prompt
    full_messages = initial_messages + history + [HumanMessage(content=prompt)]

    app = create_agent_graph(model_name, temperature, max_tokens)
    inputs = {"messages": full_messages}
    
    # Track usage across the stream
    final_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async for event in app.astream_events(inputs, version="v2"):
        kind = event["event"]
        
        # We only care about events from our agents, not the supervisor/router
        node_name = event.get("metadata", {}).get("langgraph_node", "")
        
        if not node_name or not node_name.startswith("agent_"):
            continue

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            content = chunk.content
            if content:
                # Handle case where content is a list (e.g. from some tool-using models)
                text_content = ""
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            text_content += part["text"]
                        elif isinstance(part, str):
                            text_content += part
                else:
                    text_content = str(content)
                
                if text_content:
                    yield {"type": "token", "content": text_content}
        
        elif kind == "on_chat_model_end":
            output = event["data"]["output"]
            # Try to extract usage
            usage = {}
            if hasattr(output, "usage_metadata") and output.usage_metadata:
                usage = output.usage_metadata
            elif hasattr(output, "response_metadata") and "token_usage" in output.response_metadata:
                 token_usage = output.response_metadata["token_usage"]
                 usage = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0)
                }
            
            if usage:
                # Accumulate if multiple agents run (though usually just one per turn)
                final_usage["input_tokens"] += usage.get("input_tokens", 0)
                final_usage["output_tokens"] += usage.get("output_tokens", 0)
                final_usage["total_tokens"] += usage.get("total_tokens", 0)

    yield {"type": "usage", "usage": final_usage}

if __name__ == "__main__":
    # Configure logging for standalone run
    logging.basicConfig(level=logging.INFO)
    print("Agent: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        result = run_agent(user_input)
        print(f"Agent: {result}")