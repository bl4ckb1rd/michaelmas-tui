import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from michaelmas.core.mcp import MCPManager


@pytest.fixture
def mcp_manager():
    return MCPManager(config_path="test_mcp_config.json")


@pytest.mark.asyncio
async def test_load_config_no_file(mcp_manager):
    config = await mcp_manager.load_config()
    assert config == {}


@pytest.mark.asyncio
async def test_load_config_with_file(mcp_manager):
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = '{"mcpServers": {"test": {}}}'
        mock_open.return_value.__enter__.return_value = mock_file
        with patch("os.path.exists", return_value=True):
            config = await mcp_manager.load_config()
            assert "mcpServers" in config


@pytest.mark.asyncio
async def test_initialize_connects_to_servers(mcp_manager):
    with patch.object(
        mcp_manager,
        "load_config",
        return_value={"mcpServers": {"test": {"command": "echo", "args": ["hello"]}}},
    ):
        with patch.object(
            mcp_manager, "connect_to_server", new_callable=AsyncMock
        ) as mock_connect:
            await mcp_manager.initialize()
            mock_connect.assert_awaited_once()


@pytest.mark.asyncio
async def test_discover_tools(mcp_manager):
    mock_session = MagicMock()

    # Mock list_tools response
    mock_tool = MagicMock()
    mock_tool.name = "my_tool"
    mock_tool.description = "desc"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {"arg1": {"type": "string"}},
    }

    mock_result = MagicMock()
    mock_result.tools = [mock_tool]

    mock_session.list_tools = AsyncMock(return_value=mock_result)

    await mcp_manager.discover_tools("test_server", mock_session)

    assert len(mcp_manager.tools) == 1
    assert mcp_manager.tools[0].name == "test_server_my_tool"
