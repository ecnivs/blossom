"""
SQLite storage backend for structured memory data.

Handles storage and retrieval of conversation sessions, turns, and semantic
memories using SQLite for efficient querying and persistence.
"""

import sqlite3
import logging
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
from contextlib import contextmanager
from ..models import ConversationTurn, ConversationSession, SemanticMemory


class SQLiteStore:
    """
    SQLite-based storage for conversation and memory data.

    Provides efficient storage and retrieval with proper indexing for:
    - Conversation sessions
    - Individual conversation turns
    - Semantic memories
    - Access tracking and metadata
    """

    def __init__(self, db_path: str):
        """
        Initialize SQLite store.

        Args:
            db_path: Path to SQLite database file
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper cleanup."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Conversation sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    summary TEXT,
                    topics TEXT,
                    total_turns INTEGER DEFAULT 0,
                    is_consolidated INTEGER DEFAULT 0,
                    consolidated_at TEXT
                )
            """)

            # Conversation turns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    speaker_name TEXT,
                    text TEXT NOT NULL,
                    language TEXT NOT NULL,
                    source TEXT NOT NULL,
                    cache_distance REAL,
                    plugin_calls TEXT,
                    importance_score REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id)
                )
            """)

            # Semantic memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    memory_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    source_sessions TEXT,
                    source_turns TEXT,
                    confidence REAL DEFAULT 0.5,
                    importance REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    last_accessed TEXT,
                    last_updated TEXT,
                    access_count INTEGER DEFAULT 0
                )
            """)

            # Create indexes for efficient querying
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_session 
                ON conversation_turns(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_timestamp 
                ON conversation_turns(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_speaker 
                ON conversation_turns(speaker)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_start_time 
                ON conversation_sessions(start_time)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type 
                ON semantic_memories(memory_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_importance 
                ON semantic_memories(importance)
            """)

            conn.commit()
            self.logger.info(f"Database initialized at {self.db_path}")

    # ==================== Session Operations ====================

    def save_session(self, session: ConversationSession) -> None:
        """Save or update a conversation session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data = session.to_dict()
            data["topics"] = json.dumps(data["topics"])
            data["is_consolidated"] = 1 if data["is_consolidated"] else 0

            cursor.execute(
                """
                INSERT OR REPLACE INTO conversation_sessions
                (session_id, start_time, end_time, summary, topics, 
                 total_turns, is_consolidated, consolidated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    data["session_id"],
                    data["start_time"],
                    data["end_time"],
                    data["summary"],
                    data["topics"],
                    data["total_turns"],
                    data["is_consolidated"],
                    data["consolidated_at"],
                ),
            )
            conn.commit()

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """Retrieve a conversation session by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM conversation_sessions WHERE session_id = ?
            """,
                (session_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            data = dict(row)
            data["topics"] = json.loads(data["topics"]) if data["topics"] else []
            data["is_consolidated"] = bool(data["is_consolidated"])

            return ConversationSession.from_dict(data)

    def get_recent_sessions(self, limit: int = 10) -> List[ConversationSession]:
        """Get most recent conversation sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM conversation_sessions 
                ORDER BY start_time DESC 
                LIMIT ?
            """,
                (limit,),
            )

            sessions = []
            for row in cursor.fetchall():
                data = dict(row)
                data["topics"] = json.loads(data["topics"]) if data["topics"] else []
                data["is_consolidated"] = bool(data["is_consolidated"])
                sessions.append(ConversationSession.from_dict(data))

            return sessions

    def get_unconsolidated_sessions(self) -> List[ConversationSession]:
        """Get sessions that haven't been consolidated yet."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM conversation_sessions 
                WHERE is_consolidated = 0 AND end_time IS NOT NULL
                ORDER BY start_time ASC
            """)

            sessions = []
            for row in cursor.fetchall():
                data = dict(row)
                data["topics"] = json.loads(data["topics"]) if data["topics"] else []
                data["is_consolidated"] = bool(data["is_consolidated"])
                sessions.append(ConversationSession.from_dict(data))

            return sessions

    # ==================== Turn Operations ====================

    def save_turn(self, turn: ConversationTurn) -> None:
        """Save a conversation turn."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data = turn.to_dict()
            data["plugin_calls"] = json.dumps(data["plugin_calls"])

            cursor.execute(
                """
                INSERT OR REPLACE INTO conversation_turns
                (turn_id, session_id, timestamp, speaker, speaker_name, text,
                 language, source, cache_distance, plugin_calls, importance_score,
                 created_at, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    data["turn_id"],
                    data["session_id"],
                    data["timestamp"],
                    data["speaker"],
                    data["speaker_name"],
                    data["text"],
                    data["language"],
                    data["source"],
                    data["cache_distance"],
                    data["plugin_calls"],
                    data["importance_score"],
                    data["created_at"],
                    data["last_accessed"],
                    data["access_count"],
                ),
            )
            conn.commit()

    def get_session_turns(self, session_id: str) -> List[ConversationTurn]:
        """Get all turns for a session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM conversation_turns 
                WHERE session_id = ? 
                ORDER BY timestamp ASC
            """,
                (session_id,),
            )

            turns = []
            for row in cursor.fetchall():
                data = dict(row)
                data["plugin_calls"] = (
                    json.loads(data["plugin_calls"]) if data["plugin_calls"] else []
                )
                turns.append(ConversationTurn.from_dict(data))

            return turns

    def get_recent_turns(self, limit: int = 20) -> List[ConversationTurn]:
        """Get most recent conversation turns."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM conversation_turns 
                ORDER BY timestamp DESC 
                LIMIT ?
            """,
                (limit,),
            )

            turns = []
            for row in cursor.fetchall():
                data = dict(row)
                data["plugin_calls"] = (
                    json.loads(data["plugin_calls"]) if data["plugin_calls"] else []
                )
                turns.append(ConversationTurn.from_dict(data))

            return list(reversed(turns))  # Return in chronological order

    def get_turn_by_id(self, turn_id: str) -> Optional[ConversationTurn]:
        """Get a specific turn by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM conversation_turns WHERE turn_id = ?
            """,
                (turn_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            data = dict(row)
            data["plugin_calls"] = (
                json.loads(data["plugin_calls"]) if data["plugin_calls"] else []
            )
            return ConversationTurn.from_dict(data)

    # ==================== Semantic Memory Operations ====================

    def save_semantic_memory(self, memory: SemanticMemory) -> None:
        """Save a semantic memory."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data = memory.to_dict()
            data["source_sessions"] = json.dumps(data["source_sessions"])
            data["source_turns"] = json.dumps(data["source_turns"])

            cursor.execute(
                """
                INSERT OR REPLACE INTO semantic_memories
                (memory_id, content, memory_type, source_sessions, source_turns,
                 confidence, importance, created_at, last_accessed, last_updated, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    data["memory_id"],
                    data["content"],
                    data["memory_type"],
                    data["source_sessions"],
                    data["source_turns"],
                    data["confidence"],
                    data["importance"],
                    data["created_at"],
                    data["last_accessed"],
                    data["last_updated"],
                    data["access_count"],
                ),
            )
            conn.commit()

    def get_semantic_memories(
        self,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        limit: Optional[int] = None,
    ) -> List[SemanticMemory]:
        """Get semantic memories with optional filtering."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM semantic_memories WHERE importance >= ?"
            params = [min_importance]

            if memory_type:
                query += " AND memory_type = ?"
                params.append(memory_type)

            query += " ORDER BY importance DESC, last_accessed DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor.execute(query, params)

            memories = []
            for row in cursor.fetchall():
                data = dict(row)
                data["source_sessions"] = (
                    json.loads(data["source_sessions"])
                    if data["source_sessions"]
                    else []
                )
                data["source_turns"] = (
                    json.loads(data["source_turns"]) if data["source_turns"] else []
                )
                memories.append(SemanticMemory.from_dict(data))

            return memories

    # ==================== Cleanup Operations ====================

    def delete_old_sessions(self, retention_days: int) -> int:
        """Delete sessions older than retention period."""
        cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get session IDs to delete
            cursor.execute(
                """
                SELECT session_id FROM conversation_sessions 
                WHERE start_time < ?
            """,
                (cutoff_date,),
            )
            session_ids = [row[0] for row in cursor.fetchall()]

            if not session_ids:
                return 0

            # Delete turns for these sessions
            placeholders = ",".join("?" * len(session_ids))
            cursor.execute(
                f"""
                DELETE FROM conversation_turns 
                WHERE session_id IN ({placeholders})
            """,
                session_ids,
            )

            # Delete sessions
            cursor.execute(
                f"""
                DELETE FROM conversation_sessions 
                WHERE session_id IN ({placeholders})
            """,
                session_ids,
            )

            conn.commit()
            self.logger.info(f"Deleted {len(session_ids)} old sessions")
            return len(session_ids)

    def update_importance_scores(self, decay_factor: float = 0.95) -> None:
        """Apply decay to importance scores."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE semantic_memories 
                SET importance = importance * ?
            """,
                (decay_factor,),
            )
            conn.commit()
