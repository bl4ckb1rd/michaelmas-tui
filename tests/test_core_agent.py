import pytest
from unittest.mock import MagicMock, AsyncMock
from michaelmas.core import agent
from langchain_core.messages import AIMessage

@pytest.mark.asyncio
async def test_run_agent_stream(mocker):
    # Mock environment
    mocker.patch.dict("os.environ", {"GOOGLE_API_KEY": "test"})
    
    # Mock create_agent_graph
    mock_app = MagicMock()
    mock_graph = mocker.patch("michaelmas.core.agent.create_agent_graph", return_value=mock_app)
    
    # Mock astream_events
    async def async_gen(*args, **kwargs):
        # Yield token chunk
        chunk = MagicMock()
        chunk.content = "Hello"
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": chunk},
            "metadata": {"langgraph_node": "agent_general"}
        }
        
        # Yield usage metadata
        output_msg = MagicMock()
        output_msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        yield {
            "event": "on_chat_model_end",
            "data": {"output": output_msg},
            "metadata": {"langgraph_node": "agent_general"}
        }

    mock_app.astream_events = async_gen
    
    # Run the generator
    results = []
    async for item in agent.run_agent("Hi"):
        results.append(item)
    
    # Assertions
    assert len(results) == 2
    assert results[0] == {"type": "token", "content": "Hello"}
    assert results[1]["type"] == "usage"
    assert results[1]["usage"]["total_tokens"] == 15

def test_create_agent_graph_gemini(mocker):
    mocker.patch.dict("os.environ", {"GOOGLE_API_KEY": "test"})
    mock_gemini = mocker.patch("michaelmas.core.agent.ChatGoogleGenerativeAI")
    mock_compile = mocker.patch("langgraph.graph.StateGraph.compile")
    
    agent.create_agent_graph("gemini-test")
    
    mock_gemini.assert_called()
    mock_compile.assert_called()

def test_create_agent_graph_ollama(mocker):
    mocker.patch.dict("os.environ", {"GOOGLE_API_KEY": "test"})
    mock_ollama = mocker.patch("michaelmas.core.agent.ChatOllama")
    mock_compile = mocker.patch("langgraph.graph.StateGraph.compile")
    
    # Clear cache to ensure new graph is created
    agent._agent_graph_cache.clear()
    
    agent.create_agent_graph("ollama:llama3")
    
    mock_ollama.assert_called_with(model="llama3", temperature=0.1)
