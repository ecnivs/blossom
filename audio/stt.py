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


class SpeechToText:
    def __init__(self) -> None:
        self.logger = logging.getLogger(__class__.__name__)
        self.shutdown_flag = threading.Event()

        self.model = Model(".models/vosk/vosk-model-en-us-0.42-gigaspeech")
        self.spk_model = SpkModel(".models/vosk/vosk-model-spk-0.4")

        self.buffer = queue.Queue()
        self.query = queue.Queue()

        self.sample_rate = 16000

        self.recognizer = KaldiRecognizer(self.model, self.sample_rate, self.spk_model)

        self.speaker = self._get_audio_vector("vince.wav")

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
        self.buffer.put(bytes(indata))

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
