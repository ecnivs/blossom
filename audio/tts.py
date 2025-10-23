from TTS.api import TTS
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
from config import config


class TextToSpeech:
    def __init__(self, workspace, sample_rate) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.workspace = workspace

        self.sample_rate = sample_rate

        self.queue = queue.Queue()

        self.coqui_model = config.tts.model_name
        self.device = config.tts.get_device()
        self.tts = TTS(
            model_name=self.coqui_model, progress_bar=config.tts.progress_bar
        ).to(self.device)

        self.voices_dir = Path(config.tts.voices_directory)
        self.voices_dir.mkdir(exist_ok=True)

        self.stt_reference = None
        self.interrupted = False
        self.current_stream = None
        self.temporarily_paused = False
        self.speak_interrupted = False

    def set_stt_reference(self, stt):
        self.stt_reference = stt

    def pause_playback(self):
        self.temporarily_paused = True
        if self.current_stream:
            sd.stop()
            self.current_stream = None

    def resume_playback(self):
        self.temporarily_paused = False

    def stop_playback(self):
        self.interrupted = True
        self.speak_interrupted = True

        sd.stop()
        if self.current_stream:
            self.current_stream = None

        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def play_wav(self, path: str):
        try:
            self.interrupted = False
            self.speak_interrupted = False

            if self.stt_reference:
                self.stt_reference.set_tts_playing(True)

            audio, samplerate = sf.read(path)

            if samplerate != self.sample_rate:
                num_samples = int(len(audio) * self.sample_rate / samplerate)
                audio = signal.resample(audio, num_samples)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = (audio * 32767).astype(np.int16)

            self._current_audio = audio
            self.current_stream = sd.play(audio, self.sample_rate)

            chunk_size = 8000
            for i in range(0, len(audio), chunk_size):
                if self.interrupted:
                    sd.stop()
                    break

                chunk = audio[i : i + chunk_size]

                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

                if not self.temporarily_paused:
                    audio_reference.update_playback(chunk)

                time.sleep(chunk_size / self.sample_rate)

            if not self.interrupted:
                sd.wait()
            else:
                sd.stop()

            audio_reference.clear()
            self.current_stream = None

            if self.stt_reference:
                self.stt_reference.set_tts_playing(False)

        except Exception as e:
            self.logger.error(f"Error playing {path}: {e}")
            if self.stt_reference:
                self.stt_reference.set_tts_playing(False)

    def speak(self, text: str, language: str = "en"):
        try:
            self.speak_interrupted = False

            sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())

            for sentence in sentences:
                if not sentence:
                    continue

                if self.speak_interrupted:
                    return

                path = self.workspace / f"{time.time_ns()}_speech.wav"

                self.tts.tts_to_file(
                    sentence,
                    file_path=path,
                    speaker_wav=self.voices_dir / f"{language}.wav",
                    language=language,
                )
                if self.speak_interrupted:
                    return

                self.queue.put(str(path))
        except Exception as e:
            raise Exception("Failed to generate speech audio") from e
