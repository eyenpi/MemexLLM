import logging
from typing import Any, Dict, List, Optional

from ..algorithms.base import BaseAlgorithm
from ..core.models import Message, MessageRole, Thread
from ..storage.base import BaseStorage
from ..utils.exceptions import (
    ResourceNotFoundError,
    StorageError,
    ThreadNotFoundError,
    ValidationError,
)

# Set up logging
logger = logging.getLogger(__name__)


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

        Raises:
            ValidationError: If storage is None
        """
        if storage is None:
            raise ValidationError("Storage cannot be None")

        self.storage = storage
        self.algorithm = algorithm
        logger.debug(
            f"Initialized HistoryManager with algorithm: {algorithm.__class__.__name__ if algorithm else None}"
        )

    def create_thread(self, metadata: Optional[Dict[str, Any]] = None) -> Thread:
        """
        Create a new conversation thread.

        Args:
            metadata (Optional[Dict[str, Any]]): Optional metadata to associate with
                the thread

        Returns:
            Thread: The newly created thread instance

        Raises:
            StorageError: If there's an error saving the thread
        """
        try:
            thread = Thread(metadata=metadata or {})
            self.storage.save_thread(thread)
            logger.debug(f"Created new thread with ID: {thread.id}")
            return thread
        except Exception as e:
            logger.error(f"Failed to create thread: {e}")
            raise StorageError(f"Failed to create thread: {e}") from e

    def get_thread(self, thread_id: str) -> Thread:
        """
        Retrieve a thread by its ID.

        If an algorithm is configured, it will determine how many messages to include
        in the thread's history. Otherwise, all stored messages are returned.

        Args:
            thread_id (str): The unique identifier of the thread

        Returns:
            Thread: The thread if found

        Raises:
            ValidationError: If thread_id is empty
            ThreadNotFoundError: If the thread does not exist
            StorageError: If there's an error retrieving the thread
        """
        if not thread_id:
            raise ValidationError("Thread ID cannot be empty")

        try:
            # Get the effective message limit from algorithm
            message_limit = self.algorithm.max_messages if self.algorithm else None

            # Get thread with optimized message limit
            thread = self.storage.get_thread(thread_id, message_limit=message_limit)
            if not thread:
                logger.error(f"Thread not found: {thread_id}")
                raise ThreadNotFoundError(f"Thread not found: {thread_id}")

            # Let algorithm process messages if configured
            if self.algorithm:
                messages = self.algorithm.get_message_window(thread.messages)
                thread.messages = messages

            logger.debug(
                f"Retrieved thread {thread_id} with {len(thread.messages)} messages"
            )
            return thread
        except ThreadNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get thread {thread_id}: {e}")
            raise StorageError(f"Failed to get thread: {e}") from e

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
        Add a new message to a thread.

        Args:
            thread_id (str): ID of the thread to add the message to
            content (str): Content of the message
            role (MessageRole): Role of the message sender
            metadata (Optional[Dict[str, Any]]): Optional metadata to associate with
                the message
            tool_calls (Optional[List[Any]]): Optional tool calls made in this message
            tool_call_id (Optional[str]): Optional ID of the tool call this message
                is responding to
            function_call (Optional[Dict[str, Any]]): Optional function call details
                if this message contains a function call
            name (Optional[str]): Optional name field for function messages

        Returns:
            Message: The newly created message

        Raises:
            ValidationError: If thread_id is empty or role is invalid
            ThreadNotFoundError: If the thread does not exist
            StorageError: If there's an error saving the message
        """
        if not thread_id:
            raise ValidationError("Thread ID cannot be empty")

        if not isinstance(content, str) and content is not None:
            raise ValidationError("Content must be a string or None")

        if role not in ("system", "user", "assistant", "tool", "function", "developer"):
            raise ValidationError(f"Invalid role: {role}")

        try:
            # Get the thread
            thread = self.storage.get_thread(thread_id)
            if not thread:
                logger.error(f"Thread not found: {thread_id}")
                raise ThreadNotFoundError(f"Thread not found: {thread_id}")

            # Create the message
            message = Message(
                role=role,
                content=content,
                metadata=metadata or {},
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                function_call=function_call,
                name=name,
            )

            # Add message to thread
            thread.messages.append(message)
            thread.updated_at = message.created_at

            # Save thread with new message
            self.storage.save_thread(thread)
            logger.debug(f"Added {role} message to thread {thread_id}")

            return message
        except ThreadNotFoundError:
            raise
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Failed to add message to thread {thread_id}: {e}")
            raise StorageError(f"Failed to add message to thread: {e}") from e

    def get_messages(self, thread_id: str) -> List[Message]:
        """
        Get all messages for a thread.

        Args:
            thread_id (str): ID of the thread to get messages for

        Returns:
            List[Message]: List of messages in the thread

        Raises:
            ThreadNotFoundError: If the thread does not exist
            ValidationError: If thread_id is empty
            StorageError: If there's an error retrieving the messages
        """
        if not thread_id:
            raise ValidationError("Thread ID cannot be empty")

        try:
            thread = self.storage.get_thread(thread_id)
            if not thread:
                logger.error(f"Thread not found: {thread_id}")
                raise ThreadNotFoundError(f"Thread not found: {thread_id}")

            return thread.messages
        except ThreadNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get messages for thread {thread_id}: {e}")
            raise StorageError(f"Failed to get messages: {e}") from e

    def list_threads(self, limit: int = 100, offset: int = 0) -> List[Thread]:
        """
        List threads with pagination.

        Args:
            limit (int): Maximum number of threads to return
            offset (int): Number of threads to skip

        Returns:
            List[Thread]: List of threads

        Raises:
            ValidationError: If limit or offset are invalid
            StorageError: If there's an error listing threads
        """
        if limit <= 0:
            raise ValidationError("Limit must be a positive integer")

        if offset < 0:
            raise ValidationError("Offset must be a non-negative integer")

        try:
            return self.storage.list_threads(limit=limit, offset=offset)
        except Exception as e:
            logger.error(f"Failed to list threads: {e}")
            raise StorageError(f"Failed to list threads: {e}") from e

    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread.

        Args:
            thread_id (str): ID of the thread to delete

        Returns:
            bool: True if deleted, False if thread not found

        Raises:
            ValidationError: If thread_id is empty
            StorageError: If there's an error deleting the thread
        """
        if not thread_id:
            raise ValidationError("Thread ID cannot be empty")

        try:
            return self.storage.delete_thread(thread_id)
        except Exception as e:
            logger.error(f"Failed to delete thread {thread_id}: {e}")
            raise StorageError(f"Failed to delete thread: {e}") from e
