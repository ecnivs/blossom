import importlib.util
import os
import traceback
import json


class PluginManager:
    def __init__(self, plugins_dir: str):
        self.plugins_dir = plugins_dir
        self.plugins = {}

    def discover_plugins(self):
        for filename in os.listdir(self.plugins_dir):
            if filename.endswith(".py") and filename not in (
                "__init__.py",
                "plugin_manager.py",
            ):
                path = os.path.join(self.plugins_dir, filename)
                module_name = filename[:-3]
                try:
                    spec = importlib.util.spec_from_file_location(module_name, path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    if hasattr(module, "register"):
                        result = module.register()
                        if isinstance(result, list):
                            for r in result:
                                self._register_plugin(r)
                        else:
                            self._register_plugin(result)
                except Exception as e:
                    print(f"[PluginManager] Failed to load {filename}: {e}")
                    traceback.print_exc()

    def _register_plugin(self, plugin_data: dict):
        name = plugin_data["name"]
        self.plugins[name] = plugin_data
        print(f"[PluginManager] Registered plugin: {name}")

    def list_plugins(self):
        return [
            {"name": n, "description": p["description"], "parameters": p["parameters"]}
            for n, p in self.plugins.items()
        ]

    def call_plugin(self, name: str, **kwargs):
        if name not in self.plugins:
            raise ValueError(f"No plugin named '{name}' loaded.")
        handler = self.plugins[name]["handler"]
        try:
            return handler(**kwargs)
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}

    def debug_dump(self):
        print(
            json.dumps({n: list(p.keys()) for n, p in self.plugins.items()}, indent=2)
        )
