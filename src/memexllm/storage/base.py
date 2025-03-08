from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.models import Thread


class BaseStorage(ABC):
    """Abstract base class for all storage backends"""

    def __init__(self, max_messages: Optional[int] = None):
        """
        Initialize storage with optional message limit.

        Args:
            max_messages (Optional[int]): Maximum number of messages to store per thread.
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
            thread (Thread): The thread to save
        """
        pass

    @abstractmethod
    def get_thread(
        self, thread_id: str, message_limit: Optional[int] = None
    ) -> Optional[Thread]:
        """
        Retrieve a thread by ID

        Args:
            thread_id (str): ID of the thread to retrieve
            message_limit (Optional[int]): Maximum number of most recent messages to return.
                If None, return all stored messages.

        Returns:
            Optional[Thread]: Thread if found, None otherwise
        """
        pass

    @abstractmethod
    def list_threads(self, limit: int = 100, offset: int = 0) -> List[Thread]:
        """
        List threads with pagination

        Args:
            limit (int): Maximum number of threads to return
            offset (int): Number of threads to skip

        Returns:
            List[Thread]: List of threads
        """
        pass

    @abstractmethod
    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread

        Args:
            thread_id (str): ID of the thread to delete

        Returns:
            bool: True if deleted, False otherwise
        """
        pass

    @abstractmethod
    def search_threads(self, query: Dict[str, Any]) -> List[Thread]:
        """
        Search for threads matching criteria

        Args:
            query (Dict[str, Any]): Search criteria

        Returns:
            List[Thread]: List of matching threads
        """
        pass
