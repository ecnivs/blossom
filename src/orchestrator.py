import logging
import os
import re
import json
from response import Gemini
from prompt import Builder
from config import config
from plugins import PluginManager


class Orchestrator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        # Use path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plugins_path = os.path.join(current_dir, "plugins")
        self.plugin_manager = PluginManager(plugins_dir=plugins_path)
        self.plugin_manager.discover_plugins()

        self.prompt = Builder()

        self.gemini = Gemini()

    def _call_plugin(self, name, **kwargs):
        if name not in self.plugin_manager.plugins:
            raise ValueError(
                f"Plugin '{name}' not found. Available: {list(self.plugin_manager.plugins.keys())}"
            )
        return self.plugin_manager.call_plugin(name, **kwargs)

    async def process(self, query):
        history = []
        max_turns = 5
        current_query = query

        for _ in range(max_turns):
            prompt = self.prompt.build("Vince", current_query)
            if history:
                prompt += "\n\nContext:\n" + "\n".join(history)

            response = await self.gemini.get_response(f"{prompt}\n\n{current_query}")

            lang, text = None, None
            for line in response.splitlines():
                if line.startswith("LANGUAGE:"):
                    lang = line[len("LANGUAGE:") :].strip()
                elif line.startswith("RESPONSE:"):
                    text = line[len("RESPONSE:") :].strip()

            if text:
                yield {
                    "LANGUAGE": lang or config.assistant.default_language,
                    "TEXT": text,
                }

            # Check for plugins
            plugin_call_found = False
            for line in response.splitlines():
                if line.startswith("PLUGIN:"):
                    plugin_call_found = True
                    match = re.match(r"PLUGIN:\s*(\w+)\((.*)\)", line)
                    if match:
                        plugin_name, args_str = match.groups()
                        try:
                            args = json.loads(args_str)
                        except json.JSONDecodeError:
                            self.logger.error(
                                f"Failed to parse plugin args: {args_str}"
                            )
                            yield {
                                "LANGUAGE": lang or config.assistant.default_language,
                                "TEXT": f"Error parsing arguments for {plugin_name}.",
                            }
                            return

                        return_to_llm_override = args.pop("_return_to_llm", None)

                        plugin_def = self.plugin_manager.plugins.get(plugin_name)
                        if not plugin_def:
                            yield {
                                "LANGUAGE": lang or config.assistant.default_language,
                                "TEXT": f"Plugin {plugin_name} not found.",
                            }
                            return

                        result = self._call_plugin(plugin_name, **args)

                        direct_tts = plugin_def.get("direct_tts", False)
                        should_return_to_llm = not direct_tts
                        if return_to_llm_override is not None:
                            should_return_to_llm = return_to_llm_override

                        if should_return_to_llm:
                            history.append(f"User: {current_query}")
                            if text:
                                history.append(f"Assistant: {text}")

                            history.append(
                                f"Plugin ({plugin_name}): {json.dumps(result)}"
                            )
                            current_query = "Continue based on plugin result."
                            break
                        else:
                            if not text:
                                response_text = result
                                if isinstance(result, (list, tuple)):
                                    response_text = " ".join(map(str, result))
                                elif isinstance(result, dict):
                                    response_text = ", ".join(
                                        f"{k}: {v}" for k, v in result.items()
                                    )

                                yield {
                                    "LANGUAGE": lang
                                    or config.assistant.default_language,
                                    "TEXT": str(response_text),
                                }
                            return

            if not plugin_call_found:
                return
