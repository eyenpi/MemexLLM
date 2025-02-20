from typing import Any, Dict, List, Optional

from ..core.models import Thread
from .base import BaseStorage


class MemoryStorage(BaseStorage):
    """In-memory storage implementation"""

    def __init__(self) -> None:
        self.threads: Dict[str, Thread] = {}

    def save_thread(self, thread: Thread) -> None:
        """Save thread to memory"""
        self.threads[thread.id] = thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """Get thread by ID"""
        return self.threads.get(thread_id)

    def list_threads(self, limit: int = 100, offset: int = 0) -> List[Thread]:
        """List threads with pagination"""
        threads = list(self.threads.values())
        return threads[offset : offset + limit]

    def delete_thread(self, thread_id: str) -> bool:
        """Delete thread by ID"""
        if thread_id in self.threads:
            del self.threads[thread_id]
            return True
        return False

    def search_threads(self, query: Dict[str, Any]) -> List[Thread]:
        """
        Basic search implementation

        Only supports exact matching on metadata fields
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
