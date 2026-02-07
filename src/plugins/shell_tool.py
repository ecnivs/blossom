import subprocess
import platform
import os


def shell_handler(command: str):
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"

        if len(output) > 2000:
            output = output[:2000] + "... [truncated]"

        return output if output.strip() else "Command executed with no output."
    except Exception as e:
        return f"Failed to execute command: {e}"


def get_context():
    return f"Operating System: {platform.system()} {platform.release()}; Shell: {os.environ.get('SHELL', 'Unknown')}; CWD: {os.getcwd()}"


def register():
    return {
        "name": "shell_tool",
        "description": "Executes shell commands on the terminal.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
            },
            "required": ["command"],
        },
        "handler": shell_handler,
        "direct_tts": False,
    }
