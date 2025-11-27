import subprocess
import logging
from langchain_core.tools import tool

@tool
def run_shell_command(command: str) -> str:
    """
    Executes a shell command on the local system.
    Use this to check system status, list files, or perform operations requested by the user.
    """
    logging.info(f"Executing shell command: {command}")
    try:
        # shell=True allows complex commands like 'ls -la | grep py'
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        
        # Limit output size to prevent context window overflow
        if len(output) > 2000:
            output = output[:2000] + "\n...(Output truncated)..."
            
        return output.strip()
    except Exception as e:
        logging.error(f"Shell command failed: {e}", exc_info=True)
        return f"Error executing command: {e}"