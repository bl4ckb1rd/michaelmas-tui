import pytest
from unittest.mock import MagicMock
from michaelmas.core import agent


@pytest.mark.asyncio
async def test_run_agent_stream(mocker):
    # Mock environment
    mocker.patch.dict("os.environ", {"GOOGLE_API_KEY": "test"})

    # Mock create_agent_graph
    mock_app = MagicMock()
    mocker.patch("michaelmas.core.agent.create_agent_graph", return_value=mock_app)

    # Mock astream_events
    async def async_gen(*args, **kwargs):
        # Yield token chunk
        chunk = MagicMock()
        chunk.content = "Hello"
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": chunk},
            "metadata": {"langgraph_node": "agent_general"},
        }

        # Yield usage metadata
        output_msg = MagicMock()
        output_msg.usage_metadata = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
        yield {
            "event": "on_chat_model_end",
            "data": {"output": output_msg},
            "metadata": {"langgraph_node": "agent_general"},
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
    mocker.patch("langgraph.graph.StateGraph.compile")

    # Clear cache to ensure new graph is created
    agent._agent_graph_cache.clear()

    agent.create_agent_graph("ollama:llama3")

    mock_ollama.assert_called_with(model="llama3", temperature=0.7, num_predict=None)


def test_check_ollama_tool_support(mocker):
    mock_ollama_show = mocker.patch("michaelmas.core.agent.ollama.show")

    # Case 1: Template has {{ .Tools }}
    mock_ollama_show.return_value = {"template": "Some template {{ .Tools }}"}
    assert agent.check_ollama_tool_support("model1") is True

    # Case 2: Known capable model name
    mock_ollama_show.side_effect = Exception("Error")
    assert agent.check_ollama_tool_support("llama3.1") is True

    # Case 3: Not capable
    assert agent.check_ollama_tool_support("dumb-model") is False


def test_create_agent_graph_with_enabled_tools(mocker):
    mocker.patch.dict("os.environ", {"GOOGLE_API_KEY": "test"})
    mocker.patch("michaelmas.core.agent.ChatGoogleGenerativeAI")
    mock_compile = mocker.patch("langgraph.graph.StateGraph.compile")

    agent._agent_graph_cache.clear()

    # Only enable web_search
    agent.create_agent_graph("gemini-test", enabled_tools=["web_search"])

    # Verify filtering logic (indirectly via coverage, hard to inspect graph nodes deeply without running it)
    # But we can check if graph construction didn't crash
    mock_compile.assert_called()


def test_create_agent_graph_openai(mocker):
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test"})
    mock_openai = mocker.patch("michaelmas.core.agent.ChatOpenAI")
    mocker.patch("langgraph.graph.StateGraph.compile")

    agent._agent_graph_cache.clear()

    agent.create_agent_graph("openai:gpt-4")

    mock_openai.assert_called_with(model="gpt-4", temperature=0.7, max_tokens=None)
