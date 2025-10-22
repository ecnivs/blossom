import logging
import random
from google.genai import Client
import asyncio
from config import config


class Gemini:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.client = Client()
        self.models = config.gemini.models

    async def get_response(self, query, model=None, max_retries=None):
        if model is None:
            model = config.gemini.default_model
        if max_retries is None:
            max_retries = config.gemini.max_retries
        current_model = self.models.get(str(model))

        if not current_model:
            self.logger.error(f"Model '{model}' not found in self.models")
            return None

        models_priority = list(self.models.values())
        start_index = models_priority.index(current_model)

        for model_index in range(start_index, len(models_priority)):
            fallback_model = models_priority[model_index]

            for attempt in range(1, max_retries + 1):
                try:
                    response = self.client.models.generate_content(
                        model=fallback_model, contents=query
                    )

                    text = getattr(response, "text", None)
                    if not text:
                        raise ValueError(f"No text in Gemini response: {response}")

                    self.logger.info(
                        f"Gemini returned (model={fallback_model}): {text}"
                    )
                    return text

                except Exception as e:
                    wait = config.gemini.retry_delay_base**attempt + random.uniform(
                        0, 1
                    )
                    self.logger.warning(
                        f"Attempt {attempt}/{max_retries} with {fallback_model} failed: {e}. Retrying in {wait:.1f}s"
                    )
                    await asyncio.sleep(wait)

            self.logger.warning(
                f"Model {fallback_model} exhausted retries, trying fallback if available"
            )

        raise Exception("All Gemini models failed after retries and fallbacks")
