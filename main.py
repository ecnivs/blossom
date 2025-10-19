import logging
from dotenv import load_dotenv
import shutil
import tempfile
from contextlib import contextmanager
import sys
import asyncio
from pathlib import Path
import threading
from audio import SpeechToText, TextToSpeech

load_dotenv()

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)s - %(message)s", force=True
)


# -------------------------------
# Temporary Workspace
# -------------------------------
@contextmanager
def new_workspace():
    """
    Context manager that creates a temporary directory and cleans it up afterward.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)


class Core:
    def __init__(self, workspace) -> None:
        self.logger = logging.getLogger(__class__.__name__)

        self.sample_rate = 16000

        self.workspace = Path(workspace)

        self.stt = None
        self.stt_thread = threading.Thread(
            target=self._start_stt_thread, daemon=True
        ).start()

        self.tts = TextToSpeech(workspace=self.workspace, sample_rate=self.sample_rate)

    def _start_stt_thread(self):
        self.stt = SpeechToText(sample_rate=self.sample_rate)
        self.stt.listen()

    def _process_queue(self):
        if not self.tts.queue.empty():
            self.tts.play_wav(self.tts.queue.get())

    async def run(self):
        query = []

        while True:
            try:
                if self.stt and not self.stt.query.empty():
                    query.append(self.stt.query.get())

                if query:
                    a = ", ".join(query)
                    self.tts.speak(f"Recognized: {a}")

                query = []

                self._process_queue()
                await asyncio.sleep(0.1)
            except RuntimeError as e:
                self.logger.critical(e)
                return

            except KeyboardInterrupt:
                self.stt.shutdown_flag.set()

                if self.stt_thread:
                    self.stt_thread.join()
                return

            except Exception as e:
                self.logger.error(e)
                return


if __name__ == "__main__":
    try:
        with new_workspace() as workspace:
            core = Core(workspace)
            asyncio.run(core.run())
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting cleanly...")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        logging.critical(f"Fatal Error: {e}")
        sys.exit(1)
