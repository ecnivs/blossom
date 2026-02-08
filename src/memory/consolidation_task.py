"""
Background task manager for memory consolidation.

Runs periodic tasks to consolidate conversation sessions and maintain
the memory system.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional


class MemoryConsolidationTask:
    """
    Background task that periodically consolidates conversation sessions.

    Runs on a schedule to extract semantic memories from completed sessions
    and perform memory maintenance operations.
    """

    def __init__(self, memory_manager, interval_minutes: int = 5):
        """
        Initialize consolidation task.

        Args:
            memory_manager: MemoryManager instance
            interval_minutes: How often to run consolidation (default: 5 minutes)
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.memory_manager = memory_manager
        self.interval_minutes = interval_minutes
        self.running = False
        self.task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the background consolidation task."""
        if self.running:
            self.logger.warning("Consolidation task already running")
            return

        self.running = True
        self.task = asyncio.create_task(self._run_loop())
        self.logger.info(
            f"Started memory consolidation task (interval: {self.interval_minutes}m)"
        )

    async def stop(self):
        """Stop the background consolidation task."""
        if not self.running:
            return

        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped memory consolidation task")

    async def _run_loop(self):
        """Main loop that runs consolidation periodically."""
        while self.running:
            try:
                await asyncio.sleep(self.interval_minutes * 60)

                if not self.running:
                    break

                await self._consolidate_sessions()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in consolidation loop: {e}")
                await asyncio.sleep(60)

    async def _consolidate_sessions(self):
        """Consolidate unconsolidated sessions."""
        try:
            sessions = await self.memory_manager.get_unconsolidated_sessions()

            if not sessions:
                self.logger.debug("No sessions to consolidate")
                return

            consolidation_delay = timedelta(minutes=5)
            sessions_to_consolidate = []

            for session in sessions:
                if session.end_time:
                    time_since_end = datetime.now() - session.end_time
                    if time_since_end >= consolidation_delay:
                        sessions_to_consolidate.append(session)
                else:
                    time_since_start = datetime.now() - session.start_time
                    if time_since_start >= timedelta(hours=1):
                        sessions_to_consolidate.append(session)

            if not sessions_to_consolidate:
                self.logger.debug(
                    f"Found {len(sessions)} unconsolidated sessions, but none are ready yet"
                )
                return

            self.logger.info(
                f"Consolidating {len(sessions_to_consolidate)} sessions..."
            )

            for session in sessions_to_consolidate:
                try:
                    await self.memory_manager.consolidate_session(session.session_id)
                except Exception as e:
                    self.logger.error(
                        f"Error consolidating session {session.session_id}: {e}"
                    )

            self.logger.info(
                f"Completed consolidation of {len(sessions_to_consolidate)} sessions"
            )

        except Exception as e:
            self.logger.error(f"Error in session consolidation: {e}")
