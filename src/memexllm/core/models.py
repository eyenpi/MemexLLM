import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

MessageRole = Literal["user", "assistant", "system"]


@dataclass
class Message:
    """
    Represents a single message in a conversation thread.

    A message contains the actual content, the role of the sender (user/assistant/system),
    and associated metadata like creation time and token count.

    Attributes:
        content (str): The actual text content of the message
        role (MessageRole): Role of the message sender ("user", "assistant", or "system")
        id (str): Unique identifier for the message (UUID)
        created_at (datetime): UTC timestamp when the message was created
        metadata (Dict[str, Any]): Additional custom metadata for the message
        token_count (Optional[int]): Number of tokens in the message content, if calculated
    """

    content: str
    role: MessageRole
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """
        Create a Message instance from a dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary containing message data with keys:
                - content: Message text content
                - role: Message sender role
                - id (optional): Message unique identifier
                - metadata (optional): Additional message metadata
                - token_count (optional): Number of tokens in content

        Returns:
            Message: A new Message instance
        """
        return cls(
            content=str(data["content"]),
            role=data["role"],
            id=data.get("id", str(uuid.uuid4())),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count"),
        )


@dataclass
class Thread:
    """
    Represents a conversation thread containing multiple messages.

    A thread maintains an ordered list of messages and associated metadata,
    tracking creation and update times.

    Attributes:
        id (str): Unique identifier for the thread (UUID)
        messages (List[Message]): Ordered list of messages in the thread
        metadata (Dict[str, Any]): Additional custom metadata for the thread
        created_at (datetime): UTC timestamp when the thread was created
        updated_at (datetime): UTC timestamp of the last modification
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_message(self, message: Message) -> None:
        """
        Add a new message to the thread.

        Updates the thread's updated_at timestamp when a message is added.

        Args:
            message (Message): The message to add to the thread
        """
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

    def get_messages(self) -> List[Message]:
        """
        Get all messages in the thread.

        Returns:
            List[Message]: List of all messages in chronological order
        """
        return self.messages

    @property
    def message_count(self) -> int:
        """
        Get the total number of messages in the thread.

        Returns:
            int: Number of messages in the thread
        """
        return len(self.messages)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the thread and its messages to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all thread data, including:
                - id: Thread identifier
                - messages: List of message dictionaries
                - metadata: Thread metadata
                - created_at: Creation timestamp (ISO format)
                - updated_at: Last update timestamp (ISO format)
        """
        return {
            "id": self.id,
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                    "metadata": msg.metadata,
                    "token_count": msg.token_count,
                }
                for msg in self.messages
            ],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Thread":
        """
        Create a Thread instance from a dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary containing thread data with keys:
                - id (optional): Thread identifier
                - messages (optional): List of message dictionaries
                - metadata (optional): Thread metadata
                - created_at (optional): Creation timestamp
                - updated_at (optional): Last update timestamp

        Returns:
            Thread: A new Thread instance with all messages restored
        """
        thread = cls(
            id=data.get("id", str(uuid.uuid4())),
            metadata=data.get("metadata", {}),
        )

        if "created_at" in data:
            thread.created_at = datetime.fromisoformat(data["created_at"])

        if "updated_at" in data:
            thread.updated_at = datetime.fromisoformat(data["updated_at"])

        for msg_data in data.get("messages", []):
            msg = Message.from_dict(msg_data)
            if "created_at" in msg_data:
                msg.created_at = datetime.fromisoformat(msg_data["created_at"])
            thread.messages.append(msg)

        return thread
