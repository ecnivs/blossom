import threading
import numpy as np


class SharedAudioReference:
    def __init__(self):
        self.current_playback = np.zeros(8000, dtype=np.int16)
        self.lock = threading.Lock()

    def update_playback(self, audio_chunk: np.ndarray):
        with self.lock:
            self.current_playback = audio_chunk.copy()

    def get_playback(self) -> np.ndarray:
        with self.lock:
            return self.current_playback.copy()

    def clear(self):
        with self.lock:
            self.current_playback = np.zeros(8000, dtype=np.int16)


audio_reference = SharedAudioReference()
