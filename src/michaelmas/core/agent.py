import os
import logging
from typing import TypedDict, Annotated, Sequence, Literal
import operator
from datetime import datetime
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

# --- Import Tools ---
from michaelmas.tools.sheets import (
    create_spreadsheet,
    get_sheet_data,
    update_sheet_data,
    append_to_sheet,
)
from michaelmas.tools.search import web_search

# Load environment variables from .env file
load_dotenv()

# Ensure the Google API key is set (only strictly needed if using Gemini)
if "GOOGLE_API_KEY" not in os.environ:
    logging.warning("GOOGLE_API_KEY not found. Gemini models will not work.")

# --- System Prompt ---
DEFAULT_SYSTEM_PROMPT = """
You are a helpful and versatile AI assistant. You can use tools to perform tasks.
You are equipped to handle Google Sheets operations and search the web for information.
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
general_tools = [web_search]

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

def create_agent_node(llm, tools):
    def agent_node(state: SupervisorState):
        agent_name = tools[0].name.split('_')[0] if tools else "General"
        logging.info(f"AGENT ({agent_name}): Running...")
        
        # Conditional tool binding: Skip for Ollama to avoid "GGGG" garbage output
        if isinstance(llm, ChatOllama):
            logging.info(f"AGENT ({agent_name}): Skipping tool binding for Ollama model.")
            model = llm
        else:
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

def create_agent_graph(model_name: str = "gemini-2.5-flash"):
    """
    Creates and compiles the agent graph for a given model name.
    Caches the compiled graph to avoid redundant work.
    """
    if model_name in _agent_graph_cache:
        return _agent_graph_cache[model_name]

    # Instantiate the appropriate LLM
    if model_name.startswith("ollama:"):
        ollama_model = model_name.split(":", 1)[1]
        logging.info(f"Using ChatOllama with model: {ollama_model}")
        # Using temperature=0.1 to avoid deterministic loops/garbage output in some quantized models
        llm = ChatOllama(model=ollama_model, temperature=0.1)
    else:
        logging.info(f"Using ChatGoogleGenerativeAI with model: {model_name}")
        llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    
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
    _agent_graph_cache[model_name] = graph
    return graph

# --- 5. Main execution function ---

def run_agent(prompt: str, model_name: str = "gemini-2.5-flash", history: list[BaseMessage] = []):
    """
    Runs the agent with a given prompt and model name.
    Returns a dictionary with the response text and usage metadata.
    """
    # Prepend current date to ensure agent has up-to-date context
    today_date_str = datetime.now().strftime("Today's date is %A, %B %d, %Y.")
    
    # Construct initial messages with system context
    initial_messages = []

    # If the history is empty or only contains a date message, add the default system prompt
    # Check if history (after date injection) is effectively empty of user/AI messages
    has_user_ai_messages = any(isinstance(msg, (HumanMessage, AIMessage)) for msg in history)
    
    if not has_user_ai_messages:
        initial_messages.append(SystemMessage(content=DEFAULT_SYSTEM_PROMPT))
    
    initial_messages.append(SystemMessage(content=today_date_str))

    # Combine initial context, history, and current prompt
    full_messages = initial_messages + history + [HumanMessage(content=prompt)]

    app = create_agent_graph(model_name)
    inputs = {"messages": full_messages}
    final_response = None
    usage_metadata = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for output in app.stream(inputs, stream_mode="values"):
        logging.debug(f"LangGraph Stream Output: {output}")
        last_message = output["messages"][-1]
        if isinstance(last_message, AIMessage):
            
            logging.debug(f"AIMessage Metadata: {last_message.response_metadata}")

            # Attempt to extract usage from standard property
            if hasattr(last_message, "usage_metadata") and last_message.usage_metadata:
                usage_metadata = last_message.usage_metadata
            
            # Fallback: Check response_metadata (Google provider specific)
            elif "token_usage" in last_message.response_metadata:
                # Structure might be {'prompt_tokens': X, 'completion_tokens': Y, 'total_tokens': Z}
                token_usage = last_message.response_metadata["token_usage"]
                usage_metadata = {
                    "input_tokens": token_usage.get("prompt_tokens", 0),
                    "output_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0)
                }
            
            if not last_message.tool_calls:
                # Handle content being a list (multimodal/structured) or string
                content = last_message.content
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                        elif isinstance(part, str):
                            text_parts.append(part)
                    final_response = "".join(text_parts)
                else:
                    final_response = str(content)

    return {
        "response": final_response if final_response else "Agent finished with no response.",
        "usage": usage_metadata
    }

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