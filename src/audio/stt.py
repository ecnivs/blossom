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
from config import config
import time
from pathlib import Path


class SpeechToText:
    def __init__(self, sample_rate, core=None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.shutdown_flag = threading.Event()
        self.core = core

        self.model = Model(config.stt.model_path)
        self.spk_model = SpkModel(config.stt.speaker_model_path)

        self.buffer = queue.Queue()
        self.sample_rate = sample_rate

        self.recognizer = KaldiRecognizer(self.model, self.sample_rate, self.spk_model)

        self.speakers = {}
        self._load_speakers(config.stt.speakers_directory)

        self.threshold = config.stt.speaker_identification_threshold

        self.echo_canceller = speexdsp.EchoCanceller_create(
            config.audio.echo_canceller.filter_length,
            config.audio.echo_canceller.frame_size,
            self.sample_rate,
        )

        self.reference_buffer = np.zeros(config.audio.blocksize, dtype=np.int16)

        self.lock = threading.Lock()

        self.is_listening = True
        self.tts_playing = False
        self.voice_interrupt_threshold = config.audio.voice_interrupt_threshold
        self.voice_activity_detected = False
        self.recognition_timeout = 0.5
        self.voice_activity_start_time = None

    def _load_speakers(self, speakers_dir: str):
        self.logger.info(f"Loading speakers from {speakers_dir}...")
        path = Path(speakers_dir)
        if not path.exists():
            self.logger.warning(f"Speakers directory {speakers_dir} not found.")
            return

        count = 0
        for file_path in path.glob("*.wav"):
            try:
                vector = self._get_audio_vector(str(file_path))
                name = file_path.stem.capitalize()  # Filename "vince.wav" -> "Vince"
                self.speakers[name] = vector
                self.logger.info(f"Loaded profile for speaker: {name}")
                count += 1
            except Exception as e:
                self.logger.error(f"Failed to load speaker {file_path}: {e}")
        self.logger.info(f"Loaded {count} speaker profiles.")

    def _get_audio_vector(self, audio_path) -> np.ndarray:
        data, _ = sf.read(audio_path)

        if data.ndim > 1:
            data = data.mean(axis=1)

        pcm_data = (data * 32767).astype(np.int16).tobytes()

        # Create a temporary recognizer for loading profiles to avoid state issues
        rec = KaldiRecognizer(self.model, self.sample_rate, self.spk_model)
        rec.AcceptWaveform(pcm_data)

        result = json.loads(rec.FinalResult())
        if "spk" not in result:
            raise ValueError("No speaker vector found in audio file")
        return np.array(result["spk"])

    def _identify_speaker(self, input_vector):
        best_score = -1.0
        best_match = "User"

        for name, vector in self.speakers.items():
            score = self._cosine_similarity(vector, input_vector)
            if score > best_score:
                best_score = score
                best_match = name

        return best_match, best_score

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Optional[sd.CallbackFlags],
    ) -> None:
        if status:
            self.logger.warning(f"Audio callback status: {status}")

        mic_data = np.frombuffer(indata, dtype=np.int16)

        self._detect_voice_activity(mic_data)

        with self.lock:
            if not self.is_listening and not self.tts_playing:
                return

        if self.tts_playing and not self.voice_activity_detected:
            return

        reference = audio_reference.get_playback()

        if len(reference) != len(mic_data):
            if len(mic_data) <= len(self.reference_buffer):
                self.reference_buffer[: len(mic_data)] = reference[: len(mic_data)]
                reference = self.reference_buffer[: len(mic_data)]
            else:
                reference = np.resize(reference, len(mic_data))

        mic_bytes = mic_data.tobytes()
        reference_bytes = reference.tobytes()
        cleaned_bytes = self.echo_canceller.process(mic_bytes, reference_bytes)

        cleaned_array = np.frombuffer(cleaned_bytes, dtype=np.int16)
        max_amplitude = np.abs(cleaned_array).max()

        if max_amplitude > config.audio.amplitude_threshold:
            self.buffer.put(cleaned_bytes)
        else:
            silence = np.zeros(len(cleaned_array), dtype=np.int16).tobytes()
            self.buffer.put(silence)

    def _cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (norm(v1) * norm(v2))

    def set_tts_playing(self, playing: bool):
        with self.lock:
            self.tts_playing = playing
            if not playing:
                self.voice_activity_detected = False

    def _detect_voice_activity(self, mic_data: np.ndarray) -> bool:
        if not self.tts_playing:
            return False

        rms_energy = np.sqrt(np.mean(mic_data.astype(np.float32) ** 2))

        if rms_energy > self.voice_interrupt_threshold:
            if not self.voice_activity_detected:
                self.voice_activity_start_time = time.time()
                if self.core and hasattr(self.core, "tts"):
                    self.core.tts.pause_playback()
            self.voice_activity_detected = True
            return True

        return False

    def _clear_buffer_and_reset(self):
        self.voice_activity_detected = False
        self.voice_activity_start_time = None
        while not self.buffer.empty():
            self.buffer.get()

    def reset_voice_interrupt_state(self):
        with self.lock:
            self.voice_activity_detected = False
            self.voice_activity_start_time = None
            self.is_listening = True
        while not self.buffer.empty():
            self.buffer.get()
        self.recognizer.Reset()

    def listen(self) -> None:
        with sd.RawInputStream(
            callback=self._audio_callback,
            channels=config.audio.channels,
            samplerate=self.sample_rate,
            blocksize=config.audio.blocksize,
            dtype=config.audio.dtype,
        ):
            self.logger.info("Listening...")

            while not self.shutdown_flag.is_set():
                data = self.buffer.get()

                if self.tts_playing and self.voice_activity_detected:
                    partial_result = json.loads(self.recognizer.PartialResult())
                    if "partial" in partial_result and partial_result["partial"]:
                        partial_text = partial_result["partial"].strip().lower()
                        if "stop" in partial_text:
                            if self.core:
                                self.core._on_voice_interrupt()
                            self._clear_buffer_and_reset()
                            self.logger.info(
                                "Voice Interrupt detected! stopped speaking"
                            )
                            continue

                if self.recognizer.AcceptWaveform(data=data):
                    result = json.loads(self.recognizer.Result())

                    if "spk" in result:
                        spk = result["spk"]

                        identified_speaker, score = self._identify_speaker(spk)
                        is_authorized = False

                        if score > self.threshold:
                            self.logger.info(
                                f"Identified speaker: {identified_speaker} (Score: {score:.2f})"
                            )
                            is_authorized = True
                        else:
                            self.logger.info(
                                f"Speaker not identified (Best: {identified_speaker}, Score: {score:.2f})"
                            )
                            if not config.stt.require_known_speaker:
                                is_authorized = True
                                identified_speaker = None  # Fallback: No specific name

                        if is_authorized:
                            if (
                                "text" in result
                                and result["text"]
                                and result["text"] != "huh"
                            ):
                                query_text = result["text"].strip()
                                self.logger.info(f"Recognized {query_text}")

                                if self.tts_playing and self.voice_activity_detected:
                                    if self.core:
                                        self.core._on_voice_interrupt()
                                    self._clear_buffer_and_reset()
                                    continue

                                if not self.tts_playing:
                                    if self.core:
                                        self.core._on_query_ready(
                                            query_text, speaker_name=identified_speaker
                                        )
                                elif (
                                    self.tts_playing
                                    and not self.voice_activity_detected
                                ):
                                    if self.core and hasattr(self.core, "tts"):
                                        self.core.tts.resume_playback()
                                elif (
                                    self.tts_playing
                                    and self.voice_activity_detected
                                    and self.voice_activity_start_time
                                ):
                                    elapsed = (
                                        time.time() - self.voice_activity_start_time
                                    )
                                    if elapsed > self.recognition_timeout:
                                        if self.core and hasattr(self.core, "tts"):
                                            self.core.tts.resume_playback()
                                        self.voice_activity_detected = False
                                        self.voice_activity_start_time = None

                                while not self.buffer.empty():
                                    self.buffer.get()
