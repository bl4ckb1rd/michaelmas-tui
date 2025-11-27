import pytest
from unittest.mock import MagicMock
from michaelmas.core.agent import create_supervisor, RouteQuery, SupervisorState
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

def test_supervisor_routing_sheets(mocker):
    mock_llm = MagicMock()
    # Mock structured output
    mock_router = MagicMock()
    mock_router.invoke.return_value = RouteQuery(destination="sheets")
    mock_llm.with_structured_output.return_value = mock_router
    
    supervisor = create_supervisor(mock_llm)
    
    state = {"messages": [HumanMessage(content="update sheet")]}
    result = supervisor(state)
    
    assert result["next"] == "agent_sheets"

def test_supervisor_routing_general(mocker):
    mock_llm = MagicMock()
    mock_router = MagicMock()
    mock_router.invoke.return_value = RouteQuery(destination="general")
    mock_llm.with_structured_output.return_value = mock_router
    
    supervisor = create_supervisor(mock_llm)
    
    state = {"messages": [HumanMessage(content="hi")]}
    result = supervisor(state)
    
    assert result["next"] == "agent_general"

def test_supervisor_routing_end_after_tool(mocker):
    mock_llm = MagicMock()
    supervisor = create_supervisor(mock_llm)
    
    # Last message is ToolMessage
    state = {"messages": [HumanMessage(content="do task"), ToolMessage(content="done", tool_call_id="1")]}
    result = supervisor(state)
    
    assert result["next"] == "end"

def test_create_agent_node(mocker):
    from michaelmas.core.agent import create_agent_node
    
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm # mocking the bind
    mock_llm.invoke.return_value = AIMessage(content="response")
    
    tools = [MagicMock()]
    tools[0].name = "test_tool"
    
    node = create_agent_node(mock_llm, tools)
    state = {"messages": [HumanMessage(content="hi")]}
    result = node(state)
    
    assert result["messages"][0].content == "response"

def test_create_tool_node(mocker):
    from michaelmas.core.agent import create_tool_node
    
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.invoke.return_value = "tool_output"
    
    node = create_tool_node([mock_tool])
    
    # Message with tool call
    msg = AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "1"}])
    state = {"messages": [msg]}
    
    result = node(state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], ToolMessage)
    assert result["messages"][0].content == "tool_output"

def test_create_tool_node_error(mocker):
    from michaelmas.core.agent import create_tool_node
    
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.invoke.side_effect = ValueError("Tool failed")
    
    node = create_tool_node([mock_tool])
    
    msg = AIMessage(content="", tool_calls=[{"name": "test_tool", "args": {}, "id": "1"}])
    state = {"messages": [msg]}
    
    result = node(state)
    assert "Error: Tool failed" in result["messages"][0].content

def test_create_tool_node_missing_tool(mocker):
    from michaelmas.core.agent import create_tool_node
    
    node = create_tool_node([]) # No tools
    
    msg = AIMessage(content="", tool_calls=[{"name": "missing_tool", "args": {}, "id": "1"}])
    state = {"messages": [msg]}
    
    result = node(state)
    assert "not found or disabled" in result["messages"][0].content
