from typing import Any, Dict, List, Optional

from ..algorithms.base import BaseAlgorithm
from ..core.models import Message, MessageRole, Thread
from ..storage.base import BaseStorage


class HistoryManager:
    """Core class for managing LLM conversation history"""

    def __init__(
        self,
        storage: BaseStorage,
        algorithm: Optional[BaseAlgorithm] = None,
    ):
        """
        Initialize the HistoryManager

        Args:
            storage: Storage backend implementation
            algorithm: History management algorithm (optional)
        """
        self.storage = storage
        self.algorithm = algorithm

    def create_thread(self, metadata: Optional[Dict[str, Any]] = None) -> Thread:
        """Create a new conversation thread"""
        thread = Thread(metadata=metadata or {})
        self.storage.save_thread(thread)
        return thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Retrieve a thread by ID"""
        return self.storage.get_thread(thread_id)

    def add_message(
        self,
        thread_id: str,
        content: str,
        role: MessageRole,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Add a message to a thread"""
        thread = self.storage.get_thread(thread_id)
        if not thread:
            raise ValueError(f"Thread with ID {thread_id} not found")

        message = Message(content=content, role=role, metadata=metadata or {})

        # Apply history management algorithm if provided
        if self.algorithm:
            self.algorithm.process_thread(thread, message)
        else:
            thread.add_message(message)

        self.storage.save_thread(thread)
        return message

    def get_messages(self, thread_id: str) -> List[Message]:
        """Get all messages in a thread"""
        thread = self.storage.get_thread(thread_id)
        if not thread:
            raise ValueError(f"Thread with ID {thread_id} not found")
        return thread.messages

    def list_threads(self, limit: int = 100, offset: int = 0) -> List[Thread]:
        """List threads with pagination"""
        return self.storage.list_threads(limit, offset)

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread"""
        return self.storage.delete_thread(thread_id)
