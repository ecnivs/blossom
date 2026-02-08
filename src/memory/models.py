"""
Core data models for the Blossom memory system.

These models represent the different types of memories stored in the system:
- ConversationTurn: Individual exchanges in a conversation
- ConversationSession: Complete conversation episodes
- SemanticMemory: Extracted knowledge and facts
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid
import numpy as np


def generate_id() -> str:
    """Generate a unique ID for memory objects."""
    return str(uuid.uuid4())


@dataclass
class ConversationTurn:
    """
    Represents a single turn in a conversation.

    A turn can be from either the user or the assistant, and includes
    metadata about how the response was generated (cache hit vs generated),
    any plugin calls made, and importance scoring.
    """

    turn_id: str
    session_id: str
    timestamp: datetime
    speaker: str  # 'user' or 'assistant'
    speaker_name: Optional[str]  # Identified speaker name (e.g., "Alice")
    text: str
    language: str
    source: str  # 'user_input', 'cache_hit', or 'generated'

    # Cache metadata (only for cache hits)
    cache_distance: Optional[float] = None

    # Plugin metadata
    plugin_calls: List[Dict[str, Any]] = field(default_factory=list)

    # Memory metadata
    embedding: Optional[np.ndarray] = None
    importance_score: float = 0.5  # 0.0 to 1.0

    # Access tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "speaker": self.speaker,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "language": self.language,
            "source": self.source,
            "cache_distance": self.cache_distance,
            "plugin_calls": self.plugin_calls,
            "importance_score": self.importance_score,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat()
            if self.last_accessed
            else None,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("last_accessed"):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        return cls(**data)


@dataclass
class ConversationSession:
    """
    Represents a complete conversation session.

    A session is a sequence of turns that form a coherent conversation.
    Sessions are automatically created and ended based on time gaps or
    explicit user actions.
    """

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    turns: List[ConversationTurn] = field(default_factory=list)

    # Session metadata
    summary: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    total_turns: int = 0

    # Consolidation tracking
    is_consolidated: bool = False
    consolidated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "summary": self.summary,
            "topics": self.topics,
            "total_turns": self.total_turns,
            "is_consolidated": self.is_consolidated,
            "consolidated_at": self.consolidated_at.isoformat()
            if self.consolidated_at
            else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Create from dictionary."""
        data = data.copy()
        data["start_time"] = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        if data.get("consolidated_at"):
            data["consolidated_at"] = datetime.fromisoformat(data["consolidated_at"])
        # Don't include turns in from_dict - they're loaded separately
        data.pop("turns", None)
        return cls(**data)


@dataclass
class SemanticMemory:
    """
    Represents extracted semantic knowledge from conversations.

    Semantic memories are facts, preferences, instructions, or contextual
    information extracted from conversation sessions. They persist across
    sessions and are used to provide relevant context.
    """

    memory_id: str
    content: str
    memory_type: str  # 'fact', 'preference', 'instruction', 'context'

    # Source tracking
    source_sessions: List[str] = field(default_factory=list)
    source_turns: List[str] = field(default_factory=list)

    # Confidence and importance
    confidence: float = 0.5  # 0.0 to 1.0
    importance: float = 0.5  # 0.0 to 1.0

    # Temporal metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # Access tracking
    access_count: int = 0

    # Vector embedding
    embedding: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "source_sessions": self.source_sessions,
            "source_turns": self.source_turns,
            "confidence": self.confidence,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat()
            if self.last_accessed
            else None,
            "last_updated": self.last_updated.isoformat()
            if self.last_updated
            else None,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticMemory":
        """Create from dictionary."""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("last_accessed"):
            data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
        if data.get("last_updated"):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)

    def update_access(self):
        """Update access tracking."""
        self.last_accessed = datetime.now()
        self.access_count += 1
