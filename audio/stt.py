import threading
from vosk import Model, KaldiRecognizer, SpkModel
import queue
import logging
import numpy as np
from typing import Any, Optional
import sounddevice as sd
import json
from numpy.linalg import norm
import soundfile as sf
from .shared_audio import audio_reference
import speexdsp


class SpeechToText:
    def __init__(self, sample_rate) -> None:
        self.logger = logging.getLogger(__class__.__name__)
        self.shutdown_flag = threading.Event()

        self.model = Model(".models/vosk/vosk-model-en-us-0.42-gigaspeech")
        self.spk_model = SpkModel(".models/vosk/vosk-model-spk-0.4")

        self.buffer = queue.Queue()
        self.query = queue.Queue()

        self.sample_rate = sample_rate

        self.recognizer = KaldiRecognizer(self.model, self.sample_rate, self.spk_model)

        self.speaker = self._get_audio_vector("vince.wav")

        self.echo_canceller = speexdsp.EchoCanceller_create(
            8000, 2048, self.sample_rate
        )

    def _get_audio_vector(self, audio_path) -> np.ndarray:
        data, _ = sf.read(audio_path)

        if data.ndim > 1:
            data = data.mean(axis=1)

        pcm_data = (data * 32767).astype(np.int16).tobytes()
        self.recognizer.AcceptWaveform(pcm_data)

        result = json.loads(self.recognizer.FinalResult())
        return result["spk"]

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

        if max_amplitude > 500:
            self.buffer.put(cleaned_bytes)
        else:
            silence = np.zeros(len(cleaned_array), dtype=np.int16).tobytes()
            self.buffer.put(silence)

    def _consine_similarity(self, data):
        consine_similarity = np.dot(self.speaker, data) / (
            norm(self.speaker) * norm(data)
        )
        self.logger.info(f"consine_similarity: {consine_similarity}")
        return consine_similarity

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

                    if "spk" in result:
                        spk = result["spk"]
                        if self._consine_similarity(spk) > 0:  # for testing
                            if "text" in result and result["text"]:
                                text = result["text"]
                                self.logger.info(f"Recognized: {text}")
                                self.query.put(text)
                        else:
                            self.logger.warning("Unrecognized speaker: Ignored")


if __name__ == "__main__":
    stt = SpeechToText()
    stt.listen()
