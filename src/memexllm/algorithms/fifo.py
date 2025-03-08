"""FIFO algorithm for conversation history management."""

from typing import List

from ..core.models import Message, Thread
from .base import BaseAlgorithm


class FIFOAlgorithm(BaseAlgorithm):
    """
    First-In-First-Out (FIFO) algorithm for conversation history management.

    This algorithm maintains a fixed-size window of messages by removing the oldest
    messages when the thread exceeds the maximum size. This helps manage memory usage
    and keeps conversations focused on recent context.

    Attributes:
        max_messages (int): Maximum number of messages to retain in a thread.
            When exceeded, older messages are removed.

    Example:
        ```python
        # Create FIFO algorithm that keeps last 50 messages
        algorithm = FIFOAlgorithm(max_messages=50)

        # When processing a thread with 51 messages, the oldest message
        # will be removed after adding the new one
        ```
    """

    def __init__(self, max_messages: int = 100):
        """
        Initialize the FIFO algorithm with specified capacity.

        Args:
            max_messages (int): Maximum number of messages to keep in a thread.
                Defaults to 100. Must be greater than 0.

        Raises:
            ValueError: If max_messages is less than or equal to 0
        """
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than 0")
        # Unlike base class, FIFO algorithm always requires a max_messages value
        super().__init__(max_messages=max_messages)
        # Ensure max_messages is treated as a non-optional int for type checking
        assert self.max_messages is not None
        self._max_messages: int = self.max_messages

    def process_thread(self, thread: Thread, new_message: Message) -> None:
        """
        Process a thread by adding a new message and trimming old messages if necessary.

        This method:
        1. Adds the new message to the thread
        2. If the thread length exceeds max_messages, removes oldest messages to maintain size

        Args:
            thread (Thread): The conversation thread to process
            new_message (Message): The new message to add to the thread
        """
        # Add the new message
        thread.add_message(new_message)

        # Get the window of messages to keep
        thread.messages = self.get_message_window(thread.messages)

    def get_message_window(self, messages: List[Message]) -> List[Message]:
        """
        Get the window of messages to use for context.

        Returns the most recent messages up to max_messages.

        Args:
            messages (List[Message]): Complete list of messages in the thread

        Returns:
            List[Message]: List of most recent messages up to max_messages
        """
        if len(messages) > self._max_messages:
            return messages[-self._max_messages :]
        return messages
