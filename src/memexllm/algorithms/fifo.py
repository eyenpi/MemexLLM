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
            max_messages (int, optional): Maximum number of messages to keep in a thread.
                Defaults to 100. Must be greater than 0.

        Raises:
            ValueError: If max_messages is less than or equal to 0
        """
        if max_messages <= 0:
            raise ValueError("max_messages must be greater than 0")
        self.max_messages = max_messages

    def process_thread(self, thread: Thread, new_message: Message) -> None:
        """
        Process a thread by adding a new message and trimming old messages if necessary.

        This method:
        1. Adds the new message to the thread
        2. If the thread length exceeds max_messages, removes oldest messages to maintain size

        Args:
            thread (Thread): The conversation thread to process
            new_message (Message): The new message to add to the thread

        Example:
            ```python
            # With max_messages=2:
            # Initial thread: [msg1]
            # After adding msg2: [msg1, msg2]
            # After adding msg3: [msg2, msg3]  # msg1 is removed
            ```
        """
        # Add the new message
        thread.add_message(new_message)

        # Trim old messages if we exceed the maximum
        if len(thread.messages) > self.max_messages:
            excess = len(thread.messages) - self.max_messages
            thread.messages = thread.messages[excess:]
