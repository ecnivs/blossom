import logging
import os
import re
import json
from datetime import datetime
from response import Gemini
from prompt import Builder
from config import config
from plugins import PluginManager
from cache.manager import SemanticCacheManager
from memory import MemoryManager, ConversationTurn, generate_id


class Orchestrator:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        plugins_path = os.path.join(current_dir, "plugins")
        self.plugin_manager = PluginManager(plugins_dir=plugins_path)
        self.plugin_manager.discover_plugins()

        self.prompt = Builder()
        self.gemini = Gemini()
        self.cache_manager = SemanticCacheManager()

        self.memory_manager = MemoryManager(
            sqlite_path=config.memory.storage_sqlite_path,
            vector_path=config.memory.storage_vector_db_path,
            working_memory_size=config.memory.working_memory_max_turns,
            enabled=config.memory.enabled,
        )
        self.current_session_id = None

    def _call_plugin(self, name, **kwargs):
        if name not in self.plugin_manager.plugins:
            raise ValueError(
                f"Plugin '{name}' not found. Available: {list(self.plugin_manager.plugins.keys())}"
            )
        return self.plugin_manager.call_plugin(name, **kwargs)

    async def process(self, query, speaker_name=None):
        history = []
        max_turns = 5
        current_query = query

        if (
            not self.current_session_id
            or self.memory_manager.should_start_new_session()
        ):
            self.current_session_id = await self.memory_manager.start_session()

        user_turn_id = generate_id()
        user_turn = ConversationTurn(
            turn_id=user_turn_id,
            session_id=self.current_session_id,
            timestamp=datetime.now(),
            speaker="user",
            speaker_name=speaker_name,
            text=query,
            language=config.assistant.default_language,
            source="user_input",
            plugin_calls=[],
            importance_score=0.5,
        )
        await self.memory_manager.add_turn(user_turn)

        recent_turns = await self.memory_manager.get_working_context(
            max_turns=config.memory.working_memory_max_turns
        )
        chat_history = [turn for turn in recent_turns if turn.turn_id != user_turn_id]

        self.logger.info(
            f"Using {len(chat_history)} turns from current session as context"
        )

        for _ in range(max_turns):
            if not history and _ == 0:
                cached_response = self.cache_manager.get(current_query)
                if cached_response:
                    self.logger.info("Serving from cache.")

                    # Extract response text based on return type
                    if isinstance(cached_response, dict):
                        response_text = cached_response.get("text", cached_response)
                        cache_distance = cached_response.get("distance", 0.0)
                    else:
                        response_text = cached_response
                        cache_distance = 0.0

                    # Record assistant turn from cache
                    assistant_turn = ConversationTurn(
                        turn_id=generate_id(),
                        session_id=self.current_session_id,
                        timestamp=datetime.now(),
                        speaker="assistant",
                        speaker_name=None,
                        text=response_text,
                        language=config.assistant.default_language,
                        source="cache_hit",
                        cache_distance=cache_distance,
                        plugin_calls=[],
                        importance_score=0.4,
                    )
                    await self.memory_manager.add_turn(assistant_turn)

                    yield {
                        "LANGUAGE": config.assistant.default_language,
                        "TEXT": response_text,
                    }
                    return

            prompt = self.prompt.build(speaker_name, current_query, chat_history)
            if history:
                prompt += "\n\nContext:\n" + "\n".join(history)

            response = await self.gemini.get_response(f"{prompt}\n\n{current_query}")

            lang, text, cacheable = None, None, True
            for line in response.splitlines():
                if line.startswith("LANGUAGE:"):
                    lang = line[len("LANGUAGE:") :].strip()
                elif line.startswith("RESPONSE:"):
                    text = line[len("RESPONSE:") :].strip()
                elif line.startswith("CACHEABLE:"):
                    cacheable_str = line[len("CACHEABLE:") :].strip().lower()
                    cacheable = cacheable_str in ["true", "yes", "1"]

            if text:
                # Record assistant turn from generation
                assistant_turn = ConversationTurn(
                    turn_id=generate_id(),
                    session_id=self.current_session_id,
                    timestamp=datetime.now(),
                    speaker="assistant",
                    speaker_name=None,
                    text=text,
                    language=lang or config.assistant.default_language,
                    source="generated",
                    cache_distance=None,
                    plugin_calls=[],
                    importance_score=0.5,
                )
                await self.memory_manager.add_turn(assistant_turn)

                yield {
                    "LANGUAGE": lang or config.assistant.default_language,
                    "TEXT": text,
                }

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

            if text and not plugin_call_found and not history:
                self.cache_manager.add(current_query, text, cacheable=cacheable)

            if not plugin_call_found:
                return
