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

        If an algorithm is configured, it will determine how many messages to include
        in the thread's history. Otherwise, all stored messages are returned.

        Args:
            thread_id (str): The unique identifier of the thread

        Returns:
            Optional[Thread]: The thread if found, None otherwise
        """
        # Get the effective message limit from algorithm
        message_limit = self.algorithm.max_messages if self.algorithm else None

        # Get thread with optimized message limit
        thread = self.storage.get_thread(thread_id, message_limit=message_limit)
        if not thread:
            return None

        # Let algorithm process messages if configured
        if self.algorithm:
            messages = self.algorithm.get_message_window(thread.messages)
            thread.messages = messages

        return thread

    def add_message(
        self,
        thread_id: str,
        content: str,
        role: MessageRole,
        metadata: Optional[Dict[str, Any]] = None,
        tool_calls: Optional[List[Any]] = None,
        tool_call_id: Optional[str] = None,
        function_call: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> Message:
        """
        Add a message to an existing thread.

        This method will:
        1. Store the full message history in storage
        2. Apply the algorithm only for context management
        3. Keep storage and algorithm concerns separate

        Args:
            thread_id (str): ID of the thread to add the message to
            content (str): The message content
            role (MessageRole): Role of the message sender (e.g., user, assistant)
            metadata (Optional[Dict[str, Any]]): Optional metadata to associate with
                the message
            tool_calls (Optional[List[Any]]): Tool calls made in this message
            tool_call_id (Optional[str]): ID of the tool call this message is responding to
            function_call (Optional[Dict[str, Any]]): Function call details if this message contains a function call
            name (Optional[str]): Name field for function messages

        Returns:
            Message: The newly created message instance

        Raises:
            ValueError: If the specified thread_id does not exist
        """
        # Get full thread history
        thread = self.storage.get_thread(thread_id)
        if not thread:
            raise ValueError(f"Thread with ID {thread_id} not found")

        # Create the new message
        message = Message(
            content=content,
            role=role,
            metadata=metadata or {},
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            function_call=function_call,
            name=name,
        )

        # Add message to storage (this will handle storage's max_messages limit)
        thread.add_message(message)
        self.storage.save_thread(thread)

        # If there's an algorithm, apply it to a copy for context management
        if self.algorithm:
            context_thread = Thread(id=thread.id, messages=thread.messages.copy())
            self.algorithm.process_thread(context_thread, message)

        return message

    def get_messages(self, thread_id: str) -> List[Message]:
        """
        Get messages from a thread.

        If an algorithm is configured, it will determine how many messages to return
        from the thread's history. Otherwise, all stored messages are returned.

        Args:
            thread_id (str): ID of the thread to retrieve messages from

        Returns:
            List[Message]: List of messages in the thread

        Raises:
            ValueError: If the specified thread_id does not exist
        """
        thread = self.get_thread(thread_id)
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
        threads = self.storage.list_threads(limit, offset)
        if self.algorithm:
            for thread in threads:
                thread.messages = self.algorithm.get_message_window(thread.messages)
        return threads

    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread and all its messages.

        Args:
            thread_id (str): ID of the thread to delete

        Returns:
            bool: True if the thread was successfully deleted, False otherwise
        """
        return self.storage.delete_thread(thread_id)
