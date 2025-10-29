import logging
from response import Gemini
from prompt import Builder
from config import config
from plugins import PluginManager


class Orchestrator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.plugin_manager = PluginManager(plugins_dir="plugins")

        self.prompt = Builder()

        self.gemini = Gemini()

    def _call_plugin(self, name, **kwargs):
        if name not in self.plugin_manager.plugins:
            raise ValueError(
                f"Plugin '{name}' not found. Available: {list(self.plugin_manager.plugins.keys())}"
            )
        return self.plugin_manager.call_plugin(name, **kwargs)

    async def process(self, query) -> dict:
        prompt = self.prompt.build("Vince", query)
        response = await self.gemini.get_response(f"{prompt}\n\n{query}")

        lang, text = None, None

        for line in response.splitlines():
            if line.startswith("LANGUAGE:"):
                lang = line[len("LANGUAGE:") :].strip()
            elif line.startswith("RESPONSE:"):
                text = line[len("RESPONSE:") :].strip()

        return {
            "LANGUAGE": lang or config.assistant.default_language,
            "TEXT": text,
        }
