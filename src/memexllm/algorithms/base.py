from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..core.models import Message, Thread


class BaseAlgorithm(ABC):
    """Abstract base class for history management algorithms"""

    @abstractmethod
    def process_thread(self, thread: "Thread", new_message: "Message") -> None:
        """
        Process a thread when a new message is added

        This method should modify the thread in-place if necessary
        (e.g., truncate old messages) and add the new message

        Args:
            thread: The conversation thread
            new_message: The new message being added
        """
        pass

    @abstractmethod
    def get_message_window(self, messages: List["Message"]) -> List["Message"]:
        """
        Get the window of messages to use for context

        This method determines which messages from the thread's history should
        be included in the context window for the LLM.

        Args:
            messages: Complete list of messages in the thread

        Returns:
            List of messages to include in the context window
        """
        pass
