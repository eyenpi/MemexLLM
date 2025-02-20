from typing import Optional

from ..core.models import Message, Thread
from .base import BaseAlgorithm


class FIFOAlgorithm(BaseAlgorithm):
    """
    First-In-First-Out algorithm for history management

    Keeps the N most recent messages in the thread
    """

    def __init__(self, max_messages: int = 100):
        """
        Initialize FIFO algorithm

        Args:
            max_messages: Maximum number of messages to keep
        """
        self.max_messages = max_messages

    def process_thread(self, thread: Thread, new_message: Message) -> None:
        """
        Add the new message and trim the thread if necessary

        Args:
            thread: The conversation thread
            new_message: The new message being added
        """
        # Add the new message
        thread.add_message(new_message)

        # Trim old messages if we exceed the maximum
        if len(thread.messages) > self.max_messages:
            excess = len(thread.messages) - self.max_messages
            thread.messages = thread.messages[excess:]
