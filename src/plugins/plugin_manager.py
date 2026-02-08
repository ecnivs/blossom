import logging
import importlib.util
import traceback
import json
from pathlib import Path
from typing import Dict, Any


class PluginManager:
    def __init__(self, plugins_dir: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.plugins_dir = Path(plugins_dir)
        self.plugins: Dict[str, Dict[str, Any]] = {}

    def discover_plugins(self):
        self.logger.info(f"Discovering plugins in {self.plugins_dir}")
        if not self.plugins_dir.exists():
            self.logger.warning(f"Plugin directory {self.plugins_dir} does not exist.")
            return

        for file_path in self.plugins_dir.glob("*.py"):
            filename = file_path.name
            if filename in ("__init__.py", "plugin_manager.py"):
                continue

            module_name = file_path.stem
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "register"):
                        result = module.register()
                        context_func = getattr(module, "get_context", None)

                        if isinstance(result, list):
                            for r in result:
                                if context_func:
                                    r["get_context"] = context_func
                                self._register_plugin(r)
                        else:
                            if context_func:
                                result["get_context"] = context_func
                            self._register_plugin(result)
            except Exception as e:
                self.logger.error(f"Failed to load plugin {filename}: {e}")
                self.logger.debug(traceback.format_exc())

    def _register_plugin(self, plugin_data: Dict[str, Any]):
        name = plugin_data.get("name")
        if not name:
            self.logger.warning("Plugin data missing 'name' field. Skipping.")
            return

        self.plugins[name] = plugin_data
        self.logger.info(f"Registered plugin: {name}")

    def list_plugins(self):
        return [
            {
                "name": n,
                "description": p["description"],
                "parameters": p["parameters"],
                "direct_tts": p.get("direct_tts", False),
            }
            for n, p in self.plugins.items()
        ]

    def get_all_plugin_contexts(self) -> str:
        contexts = []
        for name, plugin in self.plugins.items():
            if "get_context" in plugin:
                try:
                    context = plugin["get_context"]()
                    if context:
                        contexts.append(f"[{name} Context]: {context}")
                except Exception as e:
                    self.logger.error(f"Error getting context from {name}: {e}")
        return "\n".join(contexts)

    def call_plugin(self, name: str, **kwargs) -> Any:
        if name not in self.plugins:
            error_msg = f"No plugin named '{name}' loaded."
            self.logger.warning(error_msg)
            raise ValueError(error_msg)

        handler = self.plugins[name].get("handler")
        if not handler:
            self.logger.error(f"Plugin '{name}' has no handler.")
            return {"error": f"Plugin '{name}' has no handler."}

        try:
            return handler(**kwargs)
        except Exception as e:
            self.logger.error(f"Error executing plugin {name}: {e}")
            self.logger.debug(traceback.format_exc())
            return {"error": str(e)}

    def debug_dump(self):
        self.logger.debug(
            json.dumps({n: list(p.keys()) for n, p in self.plugins.items()}, indent=2)
        )
