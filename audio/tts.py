from TTS.api import TTS
import torch
import sounddevice as sd
import soundfile as sf
import logging
import queue
from pathlib import Path
import re
import time


class TextToSpeech:
    def __init__(self, workspace) -> None:
        self.logger = logging.getLogger(__class__.__name__)
        self.workspace = workspace

        self.queue = queue.Queue()

        self.coqui_model = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tts = TTS(model_name=self.coqui_model, progress_bar=True).to(self.device)

        self.voices_dir = Path(".voices")
        self.voices_dir.mkdir(exist_ok=True)

    def play_wav(self, path: str):
        try:
            audio, samplerate = sf.read(path)
            sd.play(audio, samplerate)
            sd.wait()
        except Exception as e:
            self.logger.error(f"Error playing {path}: {e}")

    def speak(self, text: str, language: str = "en"):
        try:
            sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
            for sentence in sentences:
                if not sentence:
                    continue
                path = self.workspace / f"{time.time_ns()}_speech.wav"

                self.tts.tts_to_file(
                    sentence,
                    file_path=path,
                    speaker_wav=self.voices_dir / f"{language}.wav",
                    language=language,
                )
                self.queue.put(str(path))
        except Exception as e:
            raise Exception("Failed to generate speech audio") from e
