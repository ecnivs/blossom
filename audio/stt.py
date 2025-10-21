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
from pyannote.audio import Model as EmbeddingModel, Inference
import os
from pathlib import Path
import torch

# TODO: Try improve performance, else switch back to vosk spk model


class SpeechToText:
    def __init__(self, sample_rate) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.sample_rate = sample_rate
        self.query = None
        self.threshold = 0.00

        self.buffer = queue.Queue()

        self.shutdown_flag = threading.Event()
        self.lock = threading.Lock()

        self.model = Model(".models/vosk/vosk-model-small-en-us-0.15")
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

        self.embedding_model = EmbeddingModel.from_pretrained(
            "pyannote/embedding", use_auth_token=os.getenv("HF_API_KEY")
        )
        self.inference = Inference(model=self.embedding_model, window="whole")

        self.speakers_dir = Path(".speakers")
        self.embeddings = self._load_embeddings()

        self.echo_canceller = speexdsp.EchoCanceller_create(
            8000, 2048, self.sample_rate
        )

    def _load_embeddings(self):
        return {f.stem: self.inference(str(f)) for f in self.speakers_dir.glob("*.wav")}

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

        if not self.query:
            if max_amplitude > 1000:
                self.buffer.put(cleaned_bytes)
            else:
                silence = np.zeros(len(cleaned_array), dtype=np.int16).tobytes()
                self.buffer.put(silence)

    def _get_best_speaker(self, chunk: np.ndarray) -> str | None:
        sample = {
            "waveform": torch.tensor(chunk).unsqueeze(0),
            "sample_rate": self.sample_rate,
        }
        emb = self.inference(sample)

        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy().flatten()

        best_speaker = None
        best_sim = -1.0
        for name, ref in self.embeddings.items():
            if isinstance(ref, torch.Tensor):
                ref_vec = ref.detach().cpu().numpy().flatten()
            else:
                ref_vec = ref.flatten()

            sim = np.dot(ref_vec, emb) / (np.linalg.norm(ref_vec) * np.linalg.norm(emb))

            if sim > best_sim:
                best_sim = sim
                best_speaker = name

        self.logger.info(f"Similarity with {best_speaker}: {best_sim:.3f}")

        if best_sim >= self.threshold:
            return best_speaker
        return None

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
                if not self.buffer.empty():
                    data = self.buffer.get()

                    if self.recognizer.AcceptWaveform(data=data):
                        result = json.loads(self.recognizer.Result())

                        speaker = self._get_best_speaker(
                            np.frombuffer(data, dtype=np.int16).astype(np.float32)
                            / 32768
                        )
                        if speaker is None:
                            self.logger.warning(
                                "Ignored: speaker does not match any known speaker"
                            )
                            continue

                        if "text" in result and result["text"]:
                            with self.lock:
                                self.query = result["text"].strip()
                                self.logger.info(f"Recognized: {self.query}")
                            while not self.buffer.empty():
                                self.buffer.get()
