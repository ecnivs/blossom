"""
Central Memory Manager coordinating all memory operations.

The MemoryManager is the main interface for the memory system, coordinating:
- Working memory (current conversation context)
- Episodic memory (conversation sessions)
- Semantic memory (extracted knowledge)
- Memory consolidation and retrieval
"""

import logging
import asyncio
from datetime import datetime
from typing import List, Optional, Union, Dict, Any
from .models import ConversationTurn, ConversationSession, SemanticMemory, generate_id
from .storage import SQLiteStore, VectorStore


class MemoryManager:
    """
    Central coordinator for all memory operations.

    Manages the multi-tier memory architecture:
    - Working memory: Current conversation context (in-memory)
    - Episodic memory: Conversation sessions (SQLite + vectors)
    - Semantic memory: Extracted knowledge (SQLite + vectors)
    - Long-term storage: Persistent archive
    """

    def __init__(
        self,
        sqlite_path: str = ".memory/conversations.db",
        vector_path: str = ".memory/vectors",
        working_memory_size: int = 20,
        enabled: bool = True,
    ):
        """
        Initialize Memory Manager.

        Args:
            sqlite_path: Path to SQLite database
            vector_path: Path to ChromaDB persistence directory
            working_memory_size: Number of turns to keep in working memory
            enabled: Whether memory system is enabled
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enabled = enabled

        if not self.enabled:
            self.logger.info("Memory system is disabled")
            return

        # Configuration
        self.working_memory_size = working_memory_size

        # Initialize storage backends
        try:
            self.sqlite_store = SQLiteStore(sqlite_path)
            self.vector_store = VectorStore(vector_path)
            self.logger.info("Memory system initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize memory system: {e}")
            self.enabled = False
            return

        # Working memory (in-memory cache)
        self.working_memory: List[ConversationTurn] = []
        self.current_session: Optional[ConversationSession] = None
        self.current_session_id: Optional[str] = None

        # Session management
        self.session_idle_threshold_minutes = 30

    # ==================== Session Management ====================

    async def start_session(self) -> str:
        """
        Start a new conversation session.

        Returns:
            Session ID
        """
        if not self.enabled:
            return "disabled"

        session_id = generate_id()
        self.current_session = ConversationSession(
            session_id=session_id, start_time=datetime.now(), total_turns=0
        )
        self.current_session_id = session_id
        self.working_memory = []

        # Save to database
        await asyncio.to_thread(self.sqlite_store.save_session, self.current_session)

        self.logger.info(f"Started new session: {session_id}")
        return session_id

    async def end_session(self, session_id: Optional[str] = None) -> None:
        """
        End a conversation session.

        Args:
            session_id: Optional session ID (uses current if not provided)
        """
        if not self.enabled:
            return

        if session_id is None:
            session_id = self.current_session_id

        if not session_id:
            return

        # Load session from database
        session = await asyncio.to_thread(self.sqlite_store.get_session, session_id)
        if session:
            session.end_time = datetime.now()
            await asyncio.to_thread(self.sqlite_store.save_session, session)
            self.logger.info(f"Ended session: {session_id}")

        # Clear current session if it matches
        if session_id == self.current_session_id:
            self.current_session = None
            self.current_session_id = None

    def should_start_new_session(self) -> bool:
        """
        Determine if a new session should be started based on idle time.

        Returns:
            True if a new session should be started
        """
        if not self.enabled or not self.working_memory:
            return False

        last_turn = self.working_memory[-1]
        idle_time = datetime.now() - last_turn.timestamp

        return idle_time.total_seconds() > (self.session_idle_threshold_minutes * 60)

    # ==================== Turn Management ====================

    async def add_turn(self, turn: ConversationTurn) -> None:
        """
        Add a conversation turn to memory.

        Args:
            turn: ConversationTurn to add
        """
        if not self.enabled:
            return

        # Ensure we have an active session
        if not self.current_session_id:
            await self.start_session()
            turn.session_id = self.current_session_id

        # Add to working memory
        self.working_memory.append(turn)

        # Trim working memory if needed
        if len(self.working_memory) > self.working_memory_size:
            self.working_memory = self.working_memory[-self.working_memory_size :]

        # Save to database
        await asyncio.to_thread(self.sqlite_store.save_turn, turn)

        # Add embedding to vector store (async)
        asyncio.create_task(self._add_turn_embedding_async(turn))

        # Update session turn count
        if self.current_session:
            self.current_session.total_turns += 1
            await asyncio.to_thread(
                self.sqlite_store.save_session, self.current_session
            )

        self.logger.debug(f"Added turn: {turn.turn_id} ({turn.speaker})")

    async def _add_turn_embedding_async(self, turn: ConversationTurn) -> None:
        """Add turn embedding asynchronously."""
        try:
            await asyncio.to_thread(self.vector_store.add_turn_embedding, turn)
        except Exception as e:
            self.logger.error(f"Error adding turn embedding: {e}")

    async def get_working_context(
        self, max_turns: Optional[int] = None
    ) -> List[ConversationTurn]:
        """
        Get current working memory context.

        Args:
            max_turns: Maximum number of turns to return (uses config default if None)

        Returns:
            List of recent conversation turns
        """
        if not self.enabled:
            return []

        if max_turns is None:
            max_turns = self.working_memory_size

        return self.working_memory[-max_turns:]

    # ==================== Memory Retrieval ====================

    async def retrieve_relevant_context(
        self,
        query: str,
        max_results: int = 5,
        time_decay: bool = True,
        importance_threshold: float = 0.3,
        include_semantic: bool = True,
        include_episodic: bool = True,
        exclude_turn_ids: Optional[List[str]] = None,
    ) -> List[Union[ConversationTurn, SemanticMemory]]:
        """
        Retrieve relevant memories for a given query.

        Uses hybrid retrieval combining:
        - Semantic similarity (vector search)
        - Recency (time decay)
        - Importance scoring
        - Access frequency

        Args:
            query: Query text to search for
            max_results: Maximum number of results to return
            time_decay: Apply temporal decay to relevance scores
            importance_threshold: Minimum importance score
            include_semantic: Include semantic memories
            include_episodic: Include episodic (turn) memories
            exclude_turn_ids: List of turn IDs to exclude from results (e.g., current user turn)

        Returns:
            List of relevant memories (turns and/or semantic memories)
        """
        if not self.enabled:
            return []

        results = []

        # Search semantic memories
        if include_semantic:
            semantic_results = await self._search_semantic_memories(
                query, max_results, importance_threshold
            )
            results.extend(semantic_results)

        # Search episodic memories (conversation turns)
        if include_episodic:
            episodic_results = await self._search_episodic_memories(
                query, max_results, importance_threshold, exclude_turn_ids
            )
            results.extend(episodic_results)

        # Score and rank results
        scored_results = []
        for item in results:
            score = self._calculate_relevance_score(item, time_decay=time_decay)
            scored_results.append((score, item))

        # Sort by score and return top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_results[:max_results]]

    async def _search_semantic_memories(
        self, query: str, max_results: int, min_importance: float
    ) -> List[SemanticMemory]:
        """Search semantic memories using vector similarity."""
        try:
            # Search vector store
            vector_results = await asyncio.to_thread(
                self.vector_store.search_similar_memories,
                query,
                n_results=max_results,
                min_importance=min_importance,
            )

            # Load full memories from SQLite
            memories = []
            for memory_id, distance, metadata in vector_results:
                # Get all semantic memories and find matching one
                all_memories = await asyncio.to_thread(
                    self.sqlite_store.get_semantic_memories
                )
                for memory in all_memories:
                    if memory.memory_id == memory_id:
                        memories.append(memory)
                        break

            return memories

        except Exception as e:
            self.logger.error(f"Error searching semantic memories: {e}")
            return []

    async def _search_episodic_memories(
        self,
        query: str,
        max_results: int,
        min_importance: float,
        exclude_turn_ids: Optional[List[str]] = None,
    ) -> List[ConversationTurn]:
        """Search episodic memories (conversation turns) using vector similarity."""
        try:
            # Search vector store
            vector_results = await asyncio.to_thread(
                self.vector_store.search_similar_turns,
                query,
                n_results=max_results,
                min_importance=min_importance,
            )

            # Load full turns from SQLite
            turns = []
            for turn_id, distance, metadata in vector_results:
                # Get session turns and find matching one
                session_id = metadata.get("session_id")
                if session_id:
                    turn = await asyncio.to_thread(
                        self.sqlite_store.get_turn_by_id, turn_id
                    )

                    if turn:
                        # Skip if this turn is in the exclude list
                        if exclude_turn_ids and turn.turn_id in exclude_turn_ids:
                            continue

                        turns.append(turn)

            return turns

        except Exception as e:
            self.logger.error(f"Error searching episodic memories: {e}")
            return []

    def _calculate_relevance_score(
        self,
        item: Union[ConversationTurn, SemanticMemory],
        time_decay: bool = True,
        time_decay_halflife_days: int = 7,
    ) -> float:
        """
        Calculate relevance score for a memory item.

        Score components:
        - Importance score (40%)
        - Recency (30% if time_decay enabled)
        - Access frequency (20%)
        - Confidence (10% for semantic memories)

        Args:
            item: Memory item to score
            time_decay: Apply temporal decay
            time_decay_halflife_days: Half-life for time decay

        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0

        # Importance component (40%)
        importance = getattr(item, "importance_score", None) or getattr(
            item, "importance", 0.5
        )
        score += importance * 0.4

        # Recency component (30%)
        if time_decay:
            created_at = getattr(item, "timestamp", None) or getattr(
                item, "created_at", datetime.now()
            )
            age_days = (datetime.now() - created_at).total_seconds() / 86400
            recency_score = 0.5 ** (age_days / time_decay_halflife_days)
            score += recency_score * 0.3
        else:
            score += 0.3  # No decay, full recency score

        # Access frequency component (20%)
        access_count = getattr(item, "access_count", 0)
        frequency_score = min(access_count / 10.0, 1.0)  # Normalize to 0-1
        score += frequency_score * 0.2

        # Confidence component (10% for semantic memories)
        if isinstance(item, SemanticMemory):
            score += item.confidence * 0.1
        else:
            score += 0.1  # Full confidence for conversation turns

        return min(score, 1.0)

    # ==================== Semantic Memory Operations ====================

    async def add_semantic_memory(self, memory: SemanticMemory) -> None:
        """
        Add a semantic memory.

        Args:
            memory: SemanticMemory to add
        """
        if not self.enabled:
            return

        # Save to database
        await asyncio.to_thread(self.sqlite_store.save_semantic_memory, memory)

        # Add embedding to vector store (async)
        asyncio.create_task(self._add_memory_embedding_async(memory))

        self.logger.debug(f"Added semantic memory: {memory.memory_id}")

    async def _add_memory_embedding_async(self, memory: SemanticMemory) -> None:
        """Add memory embedding asynchronously."""
        try:
            await asyncio.to_thread(self.vector_store.add_memory_embedding, memory)
        except Exception as e:
            self.logger.error(f"Error adding memory embedding: {e}")

    # ==================== Memory Consolidation ====================

    async def consolidate_session(self, session_id: str) -> None:
        """
        Consolidate a session by extracting semantic memories.

        This is a placeholder for future implementation that will use
        an LLM to extract facts, preferences, and knowledge from the session.

        Args:
            session_id: Session to consolidate
        """
        if not self.enabled:
            return

        self.logger.info(f"Consolidation for session {session_id} (placeholder)")

        # Mark session as consolidated
        session = await asyncio.to_thread(self.sqlite_store.get_session, session_id)
        if session:
            session.is_consolidated = True
            session.consolidated_at = datetime.now()
            await asyncio.to_thread(self.sqlite_store.save_session, session)

    async def get_unconsolidated_sessions(self) -> List[ConversationSession]:
        """Get sessions that need consolidation."""
        if not self.enabled:
            return []

        return await asyncio.to_thread(self.sqlite_store.get_unconsolidated_sessions)

    # ==================== Cleanup Operations ====================

    async def prune_old_memories(self, retention_days: int = 30) -> int:
        """
        Delete old conversation sessions beyond retention period.

        Args:
            retention_days: Number of days to retain

        Returns:
            Number of sessions deleted
        """
        if not self.enabled:
            return 0

        count = await asyncio.to_thread(
            self.sqlite_store.delete_old_sessions, retention_days
        )

        self.logger.info(f"Pruned {count} old sessions")
        return count

    async def decay_importance_scores(self, decay_factor: float = 0.95) -> None:
        """
        Apply decay to importance scores of semantic memories.

        Args:
            decay_factor: Multiplicative decay factor (0.0 to 1.0)
        """
        if not self.enabled:
            return

        await asyncio.to_thread(
            self.sqlite_store.update_importance_scores, decay_factor
        )

        self.logger.debug(f"Applied importance decay: {decay_factor}")

    # ==================== Statistics ====================

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.enabled:
            return {"enabled": False}

        recent_sessions = await asyncio.to_thread(
            self.sqlite_store.get_recent_sessions, limit=10
        )

        semantic_memories = await asyncio.to_thread(
            self.sqlite_store.get_semantic_memories
        )

        return {
            "enabled": True,
            "working_memory_size": len(self.working_memory),
            "current_session_id": self.current_session_id,
            "total_recent_sessions": len(recent_sessions),
            "total_semantic_memories": len(semantic_memories),
            "current_session_turns": self.current_session.total_turns
            if self.current_session
            else 0,
        }
