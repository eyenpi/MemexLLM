from copy import deepcopy
from typing import Any, Dict, List, Optional

from ..core.models import Thread
from .base import BaseStorage


class MemoryStorage(BaseStorage):
    """
    In-memory storage implementation for conversation threads.

    This storage backend keeps all threads in memory using a dictionary,
    making it suitable for testing and development but not for production use
    as data is lost when the application restarts.

    Attributes:
        threads (Dict[str, Thread]): Dictionary mapping thread IDs to Thread objects
        max_messages (Optional[int]): Maximum number of messages to store per thread
    """

    def __init__(self, max_messages: Optional[int] = None) -> None:
        """
        Initialize an empty in-memory storage.

        Args:
            max_messages (Optional[int]): Maximum number of messages to store per thread.
                If None, store all messages.
        """
        super().__init__(max_messages=max_messages)
        self.threads: Dict[str, Thread] = {}

    def save_thread(self, thread: Thread) -> None:
        """
        Save or update a thread in memory.

        If max_messages is set, only stores the most recent messages up to max_messages.

        Args:
            thread (Thread): The thread to save
        """
        # Create a copy to avoid modifying the original thread
        thread_copy = deepcopy(thread)

        # Apply storage limit if set
        if (
            self.max_messages is not None
            and len(thread_copy.messages) > self.max_messages
        ):
            thread_copy.messages = thread_copy.messages[-self.max_messages :]

        self.threads[thread_copy.id] = thread_copy

    def get_thread(
        self, thread_id: str, message_limit: Optional[int] = None
    ) -> Optional[Thread]:
        """
        Retrieve a thread by ID from memory.

        Args:
            thread_id (str): ID of the thread to retrieve
            message_limit (Optional[int]): Maximum number of most recent messages to return.
                If None, return all stored messages.

        Returns:
            Optional[Thread]: Thread if found, None otherwise
        """
        if thread_id not in self.threads:
            return None

        # Create a copy to avoid modifying the original
        thread = deepcopy(self.threads[thread_id])

        # Apply message limit if specified
        if message_limit is not None and len(thread.messages) > message_limit:
            thread.messages = thread.messages[-message_limit:]

        return thread

    def list_threads(self, limit: int = 100, offset: int = 0) -> List[Thread]:
        """
        List threads with pagination.

        Args:
            limit (int): Maximum number of threads to return
            offset (int): Number of threads to skip

        Returns:
            List[Thread]: List of threads
        """
        threads = list(self.threads.values())
        return [deepcopy(t) for t in threads[offset : offset + limit]]

    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread from memory.

        Args:
            thread_id (str): ID of the thread to delete

        Returns:
            bool: True if deleted, False if thread not found
        """
        if thread_id in self.threads:
            del self.threads[thread_id]
            return True
        return False

    def search_threads(self, query: Dict[str, Any]) -> List[Thread]:
        """
        Search for threads matching criteria.

        This implementation supports basic metadata matching.

        Args:
            query (Dict[str, Any]): Search criteria as key-value pairs to match
                against thread metadata

        Returns:
            List[Thread]: List of matching threads
        """
        results = []

        for thread in self.threads.values():
            match = True

            # Check metadata matches if specified
            if "metadata" in query:
                for key, value in query["metadata"].items():
                    if key not in thread.metadata or thread.metadata[key] != value:
                        match = False
                        break

            # Check content match if specified
            if match and "content" in query:
                content_match = False
                search_content = query["content"].lower()
                for message in thread.messages:
                    message_content = message.content
                    if isinstance(message_content, str):
                        if search_content in message_content.lower():
                            content_match = True
                            break
                    else:
                        # For structured content, check text parts
                        for item in message_content:
                            if (
                                hasattr(item, "text")
                                and search_content in item.text.lower()
                            ):
                                content_match = True
                                break
                        if content_match:
                            break
                if not content_match:
                    match = False

            if match:
                results.append(deepcopy(thread))

        return results
