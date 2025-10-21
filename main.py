import logging
from dotenv import load_dotenv
import shutil
import tempfile
from contextlib import contextmanager
import sys
import time
from pathlib import Path
import threading
from audio import SpeechToText, TextToSpeech
from orchestrator import Orchestrator

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
        self.logger = logging.getLogger(self.__class__.__name__)

        self.sample_rate = 16000

        self.workspace = Path(workspace)

        self.stt = SpeechToText(sample_rate=self.sample_rate)
        self.tts = TextToSpeech(workspace=self.workspace, sample_rate=self.sample_rate)

        self.orchestrator = Orchestrator()

    def _thread(self, target, args=None):
        if args:
            threading.Thread(target=target, args=args, daemon=True).start()
        else:
            threading.Thread(target=target, daemon=True).start()

    def run(self):
        stt_thread = threading.Thread(target=self.stt.listen, daemon=True).start()
        tts_thread = threading.Thread(
            target=self.tts.speech_worker, daemon=True
        ).start()

        while True:
            try:
                if self.stt:
                    with self.stt.lock:
                        if self.stt.query:
                            response = self.orchestrator.process(self.stt.query)
                            self.tts.speak(
                                text=response["TEXT"], language=response.get("LANGUAGE")
                            )
                            self.stt.query = None
                time.sleep(0.1)
            except RuntimeError as e:
                self.logger.critical(e)
                return

            except KeyboardInterrupt:
                self.stt.shutdown_flag.set()
                self.tts.shutdown_flag.set()

                if stt_thread:
                    stt_thread.join()
                if tts_thread:
                    tts_thread.join()
                return

            except Exception as e:
                self.logger.error(e)
                return


if __name__ == "__main__":
    try:
        with new_workspace() as workspace:
            core = Core(workspace)
            core.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Exiting cleanly...")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        logging.critical(f"Fatal Error: {e}")
        sys.exit(1)
