"""Base class for history management algorithms."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ..core.models import Message, Thread


class BaseAlgorithm(ABC):
    """Abstract base class for history management algorithms"""

    def __init__(self, max_messages: Optional[int] = None):
        """
        Initialize algorithm with optional message limit.

        Args:
            max_messages (Optional[int]): Maximum number of messages to include in context window.
                If None, include all messages.
        """
        self.max_messages = max_messages

    @abstractmethod
    def process_thread(self, thread: "Thread", new_message: "Message") -> None:
        """
        Process a thread when a new message is added

        This method should modify the thread in-place if necessary
        (e.g., truncate old messages) and add the new message

        Args:
            thread (Thread): The conversation thread
            new_message (Message): The new message being added
        """
        pass

    @abstractmethod
    def get_message_window(self, messages: List["Message"]) -> List["Message"]:
        """
        Get the window of messages to use for context

        This method determines which messages from the thread's history should
        be included in the context window for the LLM.

        Args:
            messages (List[Message]): Complete list of messages in the thread

        Returns:
            List[Message]: List of messages to include in the context window
        """
        pass
