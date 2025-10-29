import ollama
import logging


class Llm:
    def __init__(self, model):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model

    def get_response(self, query):
        response = ollama.generate(model=self.model, prompt=query)
        self.logger.info(f"{self.model} returned: {response['response']}")
        return response["response"]
