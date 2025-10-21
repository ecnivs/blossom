import threading
from vosk import Model, KaldiRecognizer
import queue
import logging
import numpy as np
from typing import Any, Optional
import sounddevice as sd
import json
from .shared_audio import audio_reference
import speexdsp


class SpeechToText:
    def __init__(self, sample_rate) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.sample_rate = sample_rate
        self.query: str = ""
        self.threshold = 0.20

        self.buffer = queue.Queue()

        self.shutdown_flag = threading.Event()
        self.lock = threading.Lock()

        self.model = Model(".models/vosk/vosk-model-small-en-us-0.15")
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

        self.echo_canceller = speexdsp.EchoCanceller_create(
            8000, 2048, self.sample_rate
        )

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Optional[sd.CallbackFlags],
    ) -> None:
        if status:
            self.logger.warning(f"Audio callback status: {status}")

        reference = audio_reference.get_playback()
        mic_data = np.frombuffer(indata, dtype=np.int16)

        if len(reference) != len(mic_data):
            reference = np.resize(reference, len(mic_data))

        mic_bytes = mic_data.tobytes()
        reference_bytes = reference.tobytes()

        cleaned_bytes = self.echo_canceller.process(mic_bytes, reference_bytes)

        cleaned_array = np.frombuffer(cleaned_bytes, dtype=np.int16)
        max_amplitude = np.abs(cleaned_array).max()

        if max_amplitude > 1000:
            self.buffer.put(cleaned_bytes)
        else:
            silence = np.zeros(len(cleaned_array), dtype=np.int16).tobytes()
            self.buffer.put(silence)

    def listen(self) -> None:
        self.logger.info("Listening...")

        with sd.RawInputStream(
            callback=self._audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=8000,
            dtype="int16",
        ):
            while not self.shutdown_flag.is_set():
                data = self.buffer.get()

                if self.recognizer.AcceptWaveform(data=data):
                    result = json.loads(self.recognizer.Result())

                    if "text" in result and result["text"]:
                        with self.lock:
                            self.query = result["text"].strip()
                            self.logger.info(f"Recognized: {self.query}")
                        while not self.buffer.empty():
                            self.buffer.get()
