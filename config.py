"""
Configuration management for Blossom Voice Assistant.
Provides type-safe configuration loading from YAML files using dataclasses.
"""

import yaml
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union, Any
import torch


@dataclass
class EchoCancellerConfig:
    """Configuration for echo cancellation."""

    filter_length: int = 8000
    frame_size: int = 2048


@dataclass
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = 16000
    channels: int = 1
    blocksize: int = 8000
    dtype: str = "int16"
    amplitude_threshold: int = 1000
    echo_canceller: EchoCancellerConfig = field(default_factory=EchoCancellerConfig)


@dataclass
class STTConfig:
    """Speech-to-Text configuration."""

    model_path: str = ".models/vosk/vosk-model-small-en-us-0.15"
    speaker_model_path: str = ".models/vosk/vosk-model-spk-0.4"
    speaker_reference_path: str = ".speakers/vince.wav"
    speaker_identification: bool = False
    speaker_identification_threshold: float = 0.20
    max_amplitude: int = 1000


@dataclass
class TTSConfig:
    """Text-to-Speech configuration."""

    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    voices_directory: str = ".voices"
    device: str = "auto"  # auto, cpu, cuda
    progress_bar: bool = True

    def get_device(self) -> torch.device:
        """Get the appropriate torch device based on configuration."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.device == "cuda":
            return torch.device("cuda")
        else:
            return torch.device("cpu")


@dataclass
class GeminiConfig:
    """Gemini AI configuration."""

    models: Dict[str, str] = field(
        default_factory=lambda: {"2.5": "gemini-2.5-flash", "2.0": "gemini-2.0-flash"}
    )
    default_model: str = "2.5"
    max_retries: int = 3
    retry_delay_base: int = 2


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "DEBUG"
    format: str = "%(levelname)s - %(message)s"
    force: bool = True

    def get_level(self) -> int:
        """Get the logging level as an integer."""
        return getattr(logging, self.level.upper(), logging.DEBUG)


@dataclass
class AppConfig:
    """Application configuration."""

    name: str = "Blossom"
    version: str = "0.1.0"
    description: str = "My Voice Assistant"
    workspace_temp: bool = True
    loop_delay: float = 0.1


@dataclass
class AssistantConfig:
    """Voice assistant configuration."""

    name: str = "Vince"
    role: str = "Personal Assistant"
    default_language: str = "en"
    supported_languages: List[str] = field(
        default_factory=lambda: [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "ja",
            "ko",
            "zh",
        ]
    )


@dataclass
class Config:
    """Main configuration class containing all configuration sections."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    app: AppConfig = field(default_factory=AppConfig)
    assistant: AssistantConfig = field(default_factory=AssistantConfig)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            Config instance loaded from the YAML file

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        return cls._from_dict(config_data)

    @classmethod
    def _from_dict(cls, config_data: Dict[str, Any]) -> "Config":
        """
        Create Config instance from dictionary data.

        Args:
            config_data: Dictionary containing configuration data

        Returns:
            Config instance
        """
        audio_config = AudioConfig(**config_data.get("audio", {}))
        if "echo_canceller" in config_data.get("audio", {}):
            audio_config.echo_canceller = EchoCancellerConfig(
                **config_data["audio"]["echo_canceller"]
            )

        stt_config = STTConfig(**config_data.get("stt", {}))
        tts_config = TTSConfig(**config_data.get("tts", {}))
        gemini_config = GeminiConfig(**config_data.get("gemini", {}))
        logging_config = LoggingConfig(**config_data.get("logging", {}))
        app_config = AppConfig(**config_data.get("app", {}))
        assistant_config = AssistantConfig(**config_data.get("assistant", {}))

        return cls(
            audio=audio_config,
            stt=stt_config,
            tts=tts_config,
            gemini=gemini_config,
            logging=logging_config,
            app=app_config,
            assistant=assistant_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "channels": self.audio.channels,
                "blocksize": self.audio.blocksize,
                "dtype": self.audio.dtype,
                "amplitude_threshold": self.audio.amplitude_threshold,
                "echo_canceller": {
                    "filter_length": self.audio.echo_canceller.filter_length,
                    "frame_size": self.audio.echo_canceller.frame_size,
                },
            },
            "stt": {
                "model_path": self.stt.model_path,
                "speaker_model_path": self.stt.speaker_model_path,
                "speaker_reference_path": self.stt.speaker_reference_path,
                "speaker_identification": self.stt.speaker_identification,
                "speaker_identification_threshold": self.stt.speaker_identification_threshold,
                "max_amplitude": self.stt.max_amplitude,
            },
            "tts": {
                "model_name": self.tts.model_name,
                "voices_directory": self.tts.voices_directory,
                "device": self.tts.device,
                "progress_bar": self.tts.progress_bar,
            },
            "gemini": {
                "models": self.gemini.models,
                "default_model": self.gemini.default_model,
                "max_retries": self.gemini.max_retries,
                "retry_delay_base": self.gemini.retry_delay_base,
            },
            "logging": {
                "level": self.logging.level,
                "format": self.logging.format,
                "force": self.logging.force,
            },
            "app": {
                "name": self.app.name,
                "version": self.app.version,
                "description": self.app.description,
                "workspace_temp": self.app.workspace_temp,
                "loop_delay": self.app.loop_delay,
            },
            "assistant": {
                "name": self.assistant.name,
                "role": self.assistant.role,
                "default_language": self.assistant.default_language,
                "supported_languages": self.assistant.supported_languages,
            },
        }

    def save_yaml(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            config_path: Path where to save the YAML configuration file
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


def load_config(config_path: Union[str, Path] = "config.yml") -> Config:
    """
    Load configuration from YAML file with fallback to defaults.

    Args:
        config_path: Path to the configuration file

    Returns:
        Config instance
    """
    try:
        return Config.from_yaml(config_path)
    except FileNotFoundError:
        logging.warning(f"Configuration file {config_path} not found, using defaults")
        return Config()
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {config_path}: {e}")
        logging.warning("Using default configuration")
        return Config()


config = load_config()
