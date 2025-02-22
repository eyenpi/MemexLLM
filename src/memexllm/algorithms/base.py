from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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
