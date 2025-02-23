from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.models import Thread


class BaseStorage(ABC):
    """Abstract base class for all storage backends"""

    def __init__(self, max_messages: Optional[int] = None):
        """
        Initialize storage with optional message limit.

        Args:
            max_messages: Maximum number of messages to store per thread.
                If None, store all messages.
        """
        self.max_messages = max_messages

    @abstractmethod
    def save_thread(self, thread: Thread) -> None:
        """
        Save or update a thread

        If max_messages is set, only the most recent messages up to max_messages
        will be stored.

        Args:
            thread: The thread to save
        """
        pass

    @abstractmethod
    def get_thread(
        self, thread_id: str, message_limit: Optional[int] = None
    ) -> Optional[Thread]:
        """
        Retrieve a thread by ID

        Args:
            thread_id: ID of the thread to retrieve
            message_limit: Maximum number of most recent messages to return.
                If None, return all stored messages.

        Returns:
            Thread if found, None otherwise
        """
        pass

    @abstractmethod
    def list_threads(self, limit: int = 100, offset: int = 0) -> List[Thread]:
        """
        List threads with pagination

        Args:
            limit: Maximum number of threads to return
            offset: Number of threads to skip

        Returns:
            List of threads
        """
        pass

    @abstractmethod
    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread

        Args:
            thread_id: ID of the thread to delete

        Returns:
            True if deleted, False otherwise
        """
        pass

    @abstractmethod
    def search_threads(self, query: Dict[str, Any]) -> List[Thread]:
        """
        Search for threads matching criteria

        Args:
            query: Search criteria

        Returns:
            List of matching threads
        """
        pass
