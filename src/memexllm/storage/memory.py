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
    """

    def __init__(self) -> None:
        """Initialize an empty in-memory storage."""
        self.threads: Dict[str, Thread] = {}

    def save_thread(self, thread: Thread) -> None:
        """
        Save or update a thread in memory.

        Args:
            thread (Thread): The thread to save. If a thread with the same ID
                already exists, it will be overwritten.
        """
        self.threads[thread.id] = thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """
        Retrieve a thread by its ID.

        Args:
            thread_id (str): The unique identifier of the thread to retrieve

        Returns:
            Optional[Thread]: The thread if found, None otherwise
        """
        return self.threads.get(thread_id)

    def list_threads(self, limit: int = 100, offset: int = 0) -> List[Thread]:
        """
        List threads with pagination support.

        Args:
            limit (int, optional): Maximum number of threads to return. Defaults to 100.
            offset (int, optional): Number of threads to skip. Defaults to 0.

        Returns:
            List[Thread]: List of threads, ordered by their insertion order.
                Returns an empty list if offset is greater than the number of threads.

        Example:
            ```python
            # Get first 10 threads
            first_page = storage.list_threads(limit=10)

            # Get next 10 threads
            second_page = storage.list_threads(limit=10, offset=10)
            ```
        """
        threads = list(self.threads.values())
        return threads[offset : offset + limit]

    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread from storage.

        Args:
            thread_id (str): The unique identifier of the thread to delete

        Returns:
            bool: True if the thread was found and deleted, False if it didn't exist
        """
        if thread_id in self.threads:
            del self.threads[thread_id]
            return True
        return False

    def search_threads(self, query: Dict[str, Any]) -> List[Thread]:
        """
        Search threads based on metadata fields.

        This is a basic implementation that only supports exact matching on metadata
        fields. All conditions in the query must match for a thread to be included
        in the results.

        Args:
            query (Dict[str, Any]): Search criteria as a dictionary. Currently only
                supports metadata field matching through the "metadata" key.

        Returns:
            List[Thread]: List of threads that match all search criteria

        Example:
            ```python
            # Find threads with specific metadata values
            matching_threads = storage.search_threads({
                "metadata": {
                    "user_id": "123",
                    "category": "support"
                }
            })
            ```
        """
        results = []

        for thread in self.threads.values():
            match = True

            # Check metadata matches
            for key, value in query.get("metadata", {}).items():
                if key not in thread.metadata or thread.metadata[key] != value:
                    match = False
                    break

            if match:
                results.append(thread)

        return results
