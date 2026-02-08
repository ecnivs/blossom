"""
ChromaDB vector storage backend for semantic search.

Handles embedding storage and retrieval for conversation turns and semantic
memories, enabling efficient semantic similarity search.
"""

import logging
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from ..models import ConversationTurn, SemanticMemory


class VectorStore:
    """
    ChromaDB-based vector storage for semantic search.

    Provides efficient semantic similarity search for:
    - Conversation turn embeddings
    - Semantic memory embeddings
    - Hybrid search with metadata filtering
    """

    def __init__(self, persist_directory: str):
        """
        Initialize vector store.

        Args:
            persist_directory: Path to ChromaDB persistence directory
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.persist_directory = persist_directory

        try:
            self.client = chromadb.PersistentClient(path=persist_directory)

            # Use local sentence-transformers embeddings
            self.embedding_fn = (
                embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )

            # Collection for conversation turn embeddings
            self.turns_collection = self.client.get_or_create_collection(
                name="conversation_turns",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

            # Collection for semantic memory embeddings
            self.memories_collection = self.client.get_or_create_collection(
                name="semantic_memories",
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

            self.logger.info(f"Vector store initialized at {persist_directory}")

        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {e}")
            raise

    # ==================== Turn Embeddings ====================

    def add_turn_embedding(self, turn: ConversationTurn) -> None:
        """
        Add or update a conversation turn embedding.

        Args:
            turn: ConversationTurn to embed and store
        """
        try:
            metadata = {
                "session_id": turn.session_id,
                "timestamp": turn.timestamp.isoformat(),
                "speaker": turn.speaker,
                "source": turn.source,
                "importance_score": turn.importance_score,
            }

            if turn.speaker_name:
                metadata["speaker_name"] = turn.speaker_name

            self.turns_collection.upsert(
                documents=[turn.text], ids=[turn.turn_id], metadatas=[metadata]
            )

        except Exception as e:
            self.logger.error(f"Error adding turn embedding: {e}")

    def search_similar_turns(
        self,
        query: str,
        n_results: int = 5,
        session_id: Optional[str] = None,
        min_importance: float = 0.0,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar conversation turns.

        Args:
            query: Query text to search for
            n_results: Maximum number of results
            session_id: Optional filter by session ID
            min_importance: Minimum importance score filter

        Returns:
            List of (turn_id, distance, metadata) tuples
        """
        try:
            where_filter = {"importance_score": {"$gte": min_importance}}
            if session_id:
                where_filter["session_id"] = session_id

            results = self.turns_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter if where_filter else None,
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            return [
                (
                    results["ids"][0][i],
                    results["distances"][0][i],
                    results["metadatas"][0][i],
                )
                for i in range(len(results["ids"][0]))
            ]

        except Exception as e:
            self.logger.error(f"Error searching turns: {e}")
            return []

    def get_turn_embedding(self, turn_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific turn."""
        try:
            result = self.turns_collection.get(ids=[turn_id], include=["embeddings"])

            if result["embeddings"] and result["embeddings"][0]:
                return np.array(result["embeddings"][0])
            return None

        except Exception as e:
            self.logger.error(f"Error getting turn embedding: {e}")
            return None

    # ==================== Semantic Memory Embeddings ====================

    def add_memory_embedding(self, memory: SemanticMemory) -> None:
        """
        Add or update a semantic memory embedding.

        Args:
            memory: SemanticMemory to embed and store
        """
        try:
            metadata = {
                "memory_type": memory.memory_type,
                "confidence": memory.confidence,
                "importance": memory.importance,
                "created_at": memory.created_at.isoformat(),
                "access_count": memory.access_count,
            }

            self.memories_collection.upsert(
                documents=[memory.content], ids=[memory.memory_id], metadatas=[metadata]
            )

        except Exception as e:
            self.logger.error(f"Error adding memory embedding: {e}")

    def search_similar_memories(
        self,
        query: str,
        n_results: int = 5,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        min_confidence: float = 0.0,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar semantic memories.

        Args:
            query: Query text to search for
            n_results: Maximum number of results
            memory_type: Optional filter by memory type
            min_importance: Minimum importance score filter
            min_confidence: Minimum confidence score filter

        Returns:
            List of (memory_id, distance, metadata) tuples
        """
        try:
            # Build where filter with proper $and syntax for ChromaDB
            where_conditions = []

            if min_importance > 0.0:
                where_conditions.append({"importance": {"$gte": min_importance}})
            if min_confidence > 0.0:
                where_conditions.append({"confidence": {"$gte": min_confidence}})
            if memory_type:
                where_conditions.append({"memory_type": memory_type})

            where_filter = None
            if len(where_conditions) == 1:
                where_filter = where_conditions[0]
            elif len(where_conditions) > 1:
                where_filter = {"$and": where_conditions}

            results = self.memories_collection.query(
                query_texts=[query], n_results=n_results, where=where_filter
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            return [
                (
                    results["ids"][0][i],
                    results["distances"][0][i],
                    results["metadatas"][0][i],
                )
                for i in range(len(results["ids"][0]))
            ]

        except Exception as e:
            self.logger.error(f"Error searching memories: {e}")
            return []

    def get_memory_embedding(self, memory_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific memory."""
        try:
            result = self.memories_collection.get(
                ids=[memory_id], include=["embeddings"]
            )

            if result["embeddings"] and result["embeddings"][0]:
                return np.array(result["embeddings"][0])
            return None

        except Exception as e:
            self.logger.error(f"Error getting memory embedding: {e}")
            return None

    # ==================== Batch Operations ====================

    def add_turns_batch(self, turns: List[ConversationTurn]) -> None:
        """Add multiple turns in a batch for efficiency."""
        if not turns:
            return

        try:
            documents = [turn.text for turn in turns]
            ids = [turn.turn_id for turn in turns]
            metadatas = [
                {
                    "session_id": turn.session_id,
                    "timestamp": turn.timestamp.isoformat(),
                    "speaker": turn.speaker,
                    "source": turn.source,
                    "importance_score": turn.importance_score,
                }
                for turn in turns
            ]

            self.turns_collection.upsert(
                documents=documents, ids=ids, metadatas=metadatas
            )

            self.logger.debug(f"Added {len(turns)} turns in batch")

        except Exception as e:
            self.logger.error(f"Error in batch turn addition: {e}")

    def add_memories_batch(self, memories: List[SemanticMemory]) -> None:
        """Add multiple semantic memories in a batch for efficiency."""
        if not memories:
            return

        try:
            documents = [memory.content for memory in memories]
            ids = [memory.memory_id for memory in memories]
            metadatas = [
                {
                    "memory_type": memory.memory_type,
                    "confidence": memory.confidence,
                    "importance": memory.importance,
                    "created_at": memory.created_at.isoformat(),
                    "access_count": memory.access_count,
                }
                for memory in memories
            ]

            self.memories_collection.upsert(
                documents=documents, ids=ids, metadatas=metadatas
            )

            self.logger.debug(f"Added {len(memories)} memories in batch")

        except Exception as e:
            self.logger.error(f"Error in batch memory addition: {e}")

    # ==================== Cleanup Operations ====================

    def delete_turn_embeddings(self, turn_ids: List[str]) -> None:
        """Delete turn embeddings by IDs."""
        if not turn_ids:
            return

        try:
            self.turns_collection.delete(ids=turn_ids)
            self.logger.debug(f"Deleted {len(turn_ids)} turn embeddings")
        except Exception as e:
            self.logger.error(f"Error deleting turn embeddings: {e}")

    def delete_memory_embeddings(self, memory_ids: List[str]) -> None:
        """Delete memory embeddings by IDs."""
        if not memory_ids:
            return

        try:
            self.memories_collection.delete(ids=memory_ids)
            self.logger.debug(f"Deleted {len(memory_ids)} memory embeddings")
        except Exception as e:
            self.logger.error(f"Error deleting memory embeddings: {e}")
