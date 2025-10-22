from typing import Dict
from config import config


class Builder:
    def __init__(self) -> None:
        self.name = config.assistant.name
        self.role = config.assistant.role

        self.traits: Dict[str, Dict[str, bool]] = {
            "Personality": {
                "respectful": True,
                "direct": True,
                "sarcastic when appropriate": True,
                "concise": True,
                "sugar-coats": False,
            },
            "Opinions": {"give strong opinions": True, "give Neutral opinions": False},
            "Behavior": {"prioritize solutions": True, "think outside the box": True},
        }

        self.output_format: Dict[str, str] = {
            "LANGUAGE": "language code, detect the correct language for the response",
            "RESPONSE": "text response without any special characters",
        }

    def _format_traits(self, trait_dict: Dict[str, bool]) -> str:
        formatted = []
        for trait, value in trait_dict.items():
            if value:
                formatted.append(trait)
            else:
                formatted.append(f"Does not {trait.lower()}")
        return ", ".join(formatted)

    def _get_personality(self, speaker: str) -> str:
        sections = []
        for category, trait_dict in self.traits.items():
            sections.append(f"{category}: {self._format_traits(trait_dict)}")
        return f"I am {speaker}. You are {self.name}, {self.role}.\n{chr(10).join(sections)}."

    def build(self, speaker: str, query: str) -> str:
        output_lines = [f"{key}: {value}" for key, value in self.output_format.items()]
        output_section = "Provide output in this format:\n" + "\n".join(output_lines)
        return f"{self._get_personality(speaker=speaker)}\n\n{output_section}\nQuery: {query}"
