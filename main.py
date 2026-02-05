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
from orchestrator import Orchestrator
from config import config

load_dotenv()

logging.basicConfig(
    level=config.logging.get_level(),
    format=config.logging.format,
    force=config.logging.force,
)


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

        self.sample_rate = config.audio.sample_rate

        self.workspace = Path(workspace)

        self.query_event = asyncio.Event()
        self.current_query = None
        self.shutdown_event = asyncio.Event()
        self.event_loop = None

        self.stt = None
        self.stt_thread = threading.Thread(target=self._start_stt_thread, daemon=True)
        self.stt_thread.start()

        self.tts = TextToSpeech(workspace=self.workspace, sample_rate=self.sample_rate)

        self.orchestrator = Orchestrator()
        self.voice_interrupt_event = asyncio.Event()

        self.queue_shutdown_event = threading.Event()
        self.queue_thread = threading.Thread(
            target=self._process_queue_loop, daemon=True
        )
        self.queue_thread.start()

    def _start_stt_thread(self):
        self.stt = SpeechToText(sample_rate=self.sample_rate, core=self)
        self.tts.set_stt_reference(self.stt)
        self.stt.listen()

    def _process_queue_loop(self):
        while not self.queue_shutdown_event.is_set():
            try:
                if not self.tts.queue.empty():
                    if self.stt:
                        self.stt.set_tts_playing(True)

                while not self.tts.queue.empty():
                    if self.queue_shutdown_event.is_set():
                        break
                    self.tts.play_wav(self.tts.queue.get())

                self.queue_shutdown_event.wait(0.1)

            except Exception as e:
                self.logger.error(f"Error in queue processing loop: {e}")
                self.queue_shutdown_event.wait(0.1)

    def _on_query_ready(self, query: str):
        self.current_query = query
        if self.event_loop:
            self.event_loop.call_soon_threadsafe(self.query_event.set)
        else:
            self.query_event.set()

    def _on_voice_interrupt(self):
        self.tts.stop_playback()
        if self.event_loop:
            self.event_loop.call_soon_threadsafe(self.voice_interrupt_event.set)
        else:
            self.voice_interrupt_event.set()

    async def _speak_async(self, text: str, language: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.tts.speak, text, language)

    async def run(self):
        self.event_loop = asyncio.get_running_loop()

        while not self.shutdown_event.is_set():
            try:
                _, pending = await asyncio.wait(
                    [
                        asyncio.create_task(self.query_event.wait()),
                        asyncio.create_task(self.voice_interrupt_event.wait()),
                        asyncio.create_task(self.shutdown_event.wait()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for task in pending:
                    task.cancel()

                if self.shutdown_event.is_set():
                    break

                if self.voice_interrupt_event.is_set():
                    self.voice_interrupt_event.clear()
                    if self.stt:
                        self.stt.reset_voice_interrupt_state()
                    continue

                if self.current_query:
                    async for response in self.orchestrator.process(self.current_query):
                        await self._speak_async(
                            response["TEXT"], response["LANGUAGE"].lower()
                        )

                    self.current_query = None
                    self.query_event.clear()

            except RuntimeError as e:
                self.logger.critical(e)
                break
            except Exception as e:
                self.logger.error(e)
                break

        self.queue_shutdown_event.set()
        if self.queue_thread:
            self.queue_thread.join()

        if self.stt:
            self.stt.shutdown_flag.set()
        if self.stt_thread:
            self.stt_thread.join()


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
