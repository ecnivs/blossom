import os
from typing import Dict
from config import config
from plugins import PluginManager


class Builder:
    def __init__(self) -> None:
        self.name = config.assistant.name
        self.role = config.assistant.role

        self.traits: Dict[str, Dict[str, bool]] = {
            "Personality": {
                "friendly": True,
                "respectful": True,
                "direct": True,
                "sarcastic when appropriate": True,
                "concise": True,
                "sugar-coats": False,
            },
            "Opinions": {"give strong opinions": True, "give Neutral opinions": False},
            "Behavior": {
                "think outside the box": True,
                "splits response into not too short sentences": True,
            },
        }

        self.output_format: Dict[str, str] = {
            "LANGUAGE": "language code, detect the correct language for the response",
            "RESPONSE": "text response without any special characters",
        }

        # Use path relative to the src root (parent of prompt dir)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_root = os.path.dirname(current_dir)
        plugins_path = os.path.join(src_root, "plugins")
        self.plugin_manager = PluginManager(plugins_dir=plugins_path)
        self.plugin_manager.discover_plugins()

    def _get_plugin_instructions(self) -> str:
        plugins = self.plugin_manager.list_plugins()
        if not plugins:
            return ""

        instructions = ["\nAvailable Plugins:"]
        for p in plugins:
            instructions.append(f"- {p['name']}: {p['description']}")
            instructions.append(f"  Args: {p['parameters']}")
            if p.get("direct_tts"):
                instructions.append("(Default: Direct to Output)")
            else:
                instructions.append("(Default: Return to LLM)")

        instructions.append("\nTo call a plugin, you must strictly follow this format:")
        instructions.append("PLUGIN: <plugin_name>(<json_args>)")
        instructions.append('Example: PLUGIN: some_tool({"arg": "value"})')
        instructions.append(
            "\nYou may also include a RESPONSE: line to speak to the user *before* or *while* the action is taken."
        )
        instructions.append(
            'Example:\nRESPONSE: I\'m doing that now.\nPLUGIN: some_tool({"arg": "value"})'
        )
        instructions.append(
            "\nTo force a return to LLM context (e.g. for chaining), allow the user to continue,"
        )
        instructions.append('add "_return_to_llm": true to the arguments.')

        return "\n".join(instructions)

    def _format_traits(self, trait_dict: Dict[str, bool]) -> str:
        formatted = []
        for trait, value in trait_dict.items():
            if value:
                formatted.append(trait)
            else:
                formatted.append(f"Does not {trait.lower()}")
        return ", ".join(formatted)

    def _get_personality(self, speaker: str) -> str:
        os_info = f"You are running on {config.app.os}."
        return f"I am {speaker}. You are {self.name}, {self.role}. {os_info}"

    def build(self, speaker: str, query: str) -> str:
        output_lines = [f"{key}: {value}" for key, value in self.output_format.items()]
        output_section = "Provide output in this format:\n" + "\n".join(output_lines)
        plugin_section = self._get_plugin_instructions()

        timing_instruction = (
            "\nNote: If you provide a RESPONSE, it is spoken to the user WHILE the plugin action is being performed. "
            "Speak in the present tense (e.g., 'Opening...' not 'I will open...')."
        )

        return f"{self._get_personality(speaker=speaker)}{timing_instruction}\n\n{plugin_section}\n\n{output_section}\nQuery: {query}"
