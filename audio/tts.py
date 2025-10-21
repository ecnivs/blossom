import threading
from TTS.api import TTS
import torch
import sounddevice as sd
import soundfile as sf
import logging
import queue
from pathlib import Path
import re
import time
from .shared_audio import audio_reference
import numpy as np
from scipy import signal


class TextToSpeech:
    def __init__(self, workspace, sample_rate) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.workspace = workspace
        self.sample_rate = sample_rate

        self.shutdown_flag = threading.Event()

        self.queue = queue.Queue()

        self.coqui_model = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tts = TTS(model_name=self.coqui_model, progress_bar=True).to(self.device)

        self.voices_dir = Path(".voices")
        self.voices_dir.mkdir(exist_ok=True)

    def speech_worker(self):
        while not self.shutdown_flag.set():
            if not self.queue.empty():
                self._play_wav(self.queue.get())

    def _play_wav(self, path: str):
        try:
            audio, samplerate = sf.read(path)

            if samplerate != self.sample_rate:
                num_samples = int(len(audio) * self.sample_rate / samplerate)
                audio = signal.resample(audio, num_samples)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = (audio * 32767).astype(np.int16)

            sd.play(audio, self.sample_rate)

            chunk_size = 8000
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]

                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                audio_reference.update_playback(chunk)
                self.logger.info(
                    f"Updated reference: min={chunk.min()}, max={chunk.max()}, mean={chunk.mean()}"
                )
                time.sleep(chunk_size / self.sample_rate)

            sd.wait()
            audio_reference.clear()

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
