import pytest
from unittest.mock import MagicMock, patch
from michaelmas.tools.shell import run_shell_command

def test_run_shell_command_success():
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = "Hello World"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = run_shell_command.invoke("echo Hello World")
        
        assert result == "Hello World"
        mock_run.assert_called_once()

def test_run_shell_command_with_stderr():
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = "Output"
        mock_result.stderr = "Error"
        mock_run.return_value = mock_result
        
        result = run_shell_command.invoke("some_command")
        
        assert "Output" in result
        assert "STDERR:\nError" in result

def test_run_shell_command_timeout():
    with patch("subprocess.run", side_effect=TimeoutError("Timeout")):
        result = run_shell_command.invoke("sleep 100")
        assert "Error executing command" in result
        assert "Timeout" in result

def test_run_shell_command_exception():
    with patch("subprocess.run", side_effect=Exception("Boom")):
        result = run_shell_command.invoke("boom")
        assert "Error executing command" in result
        assert "Boom" in result

def test_run_shell_command_truncation():
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.stdout = "a" * 3000
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        result = run_shell_command.invoke("long_output")
        
        assert len(result) < 3000
        assert "...(Output truncated)..." in result
