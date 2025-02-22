from typing import Any, Dict, List, Optional

from ..algorithms.base import BaseAlgorithm
from ..core.models import Message, MessageRole, Thread
from ..storage.base import BaseStorage


class HistoryManager:
    """
    Core class for managing LLM conversation history.

    The HistoryManager provides a high-level interface for managing conversation threads
    and messages, with optional support for history management algorithms.

    Attributes:
        storage (BaseStorage): Storage backend for persisting threads and messages
        algorithm (Optional[BaseAlgorithm]): Algorithm for managing conversation history
    """

    def __init__(
        self,
        storage: BaseStorage,
        algorithm: Optional[BaseAlgorithm] = None,
    ):
        """
        Initialize a new HistoryManager instance.

        Args:
            storage (BaseStorage): Storage backend implementation for persisting data
            algorithm (Optional[BaseAlgorithm]): History management algorithm for
                controlling conversation history. If None, messages are simply appended
                to threads.
        """
        self.storage = storage
        self.algorithm = algorithm

    def create_thread(self, metadata: Optional[Dict[str, Any]] = None) -> Thread:
        """
        Create a new conversation thread.

        Args:
            metadata (Optional[Dict[str, Any]]): Optional metadata to associate with
                the thread

        Returns:
            Thread: The newly created thread instance
        """
        thread = Thread(metadata=metadata or {})
        self.storage.save_thread(thread)
        return thread

    def get_thread(self, thread_id: str) -> Optional[Thread]:
        """
        Retrieve a thread by its ID.

        Args:
            thread_id (str): The unique identifier of the thread

        Returns:
            Optional[Thread]: The thread if found, None otherwise
        """
        return self.storage.get_thread(thread_id)

    def add_message(
        self,
        thread_id: str,
        content: str,
        role: MessageRole,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Add a message to an existing thread.

        This method will apply the configured history management algorithm (if any)
        before saving the message.

        Args:
            thread_id (str): ID of the thread to add the message to
            content (str): The message content
            role (MessageRole): Role of the message sender (e.g., user, assistant)
            metadata (Optional[Dict[str, Any]]): Optional metadata to associate with
                the message

        Returns:
            Message: The newly created message instance

        Raises:
            ValueError: If the specified thread_id does not exist
        """
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
        """
        Get all messages in a thread.

        Args:
            thread_id (str): ID of the thread to retrieve messages from

        Returns:
            List[Message]: List of messages in the thread

        Raises:
            ValueError: If the specified thread_id does not exist
        """
        thread = self.storage.get_thread(thread_id)
        if not thread:
            raise ValueError(f"Thread with ID {thread_id} not found")
        return thread.messages

    def list_threads(self, limit: int = 100, offset: int = 0) -> List[Thread]:
        """
        List threads with pagination support.

        Args:
            limit (int): Maximum number of threads to return (default: 100)
            offset (int): Number of threads to skip (default: 0)

        Returns:
            List[Thread]: List of thread instances
        """
        return self.storage.list_threads(limit, offset)

    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread and all its messages.

        Args:
            thread_id (str): ID of the thread to delete

        Returns:
            bool: True if the thread was successfully deleted, False otherwise
        """
        return self.storage.delete_thread(thread_id)
