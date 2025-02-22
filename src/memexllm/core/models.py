import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

MessageRole = Literal["user", "assistant", "system"]


@dataclass
class Message:
    content: str
    role: MessageRole
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a message from a dictionary"""
        return cls(
            content=str(data["content"]),
            role=data["role"],
            id=data.get("id", str(uuid.uuid4())),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count"),
        )


@dataclass
class Thread:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_message(self, message: Message) -> None:
        """Add a message to the thread"""
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

    def get_messages(self) -> List[Message]:
        """Get all messages in the thread"""
        return self.messages

    @property
    def message_count(self) -> int:
        """Get the number of messages in the thread"""
        return len(self.messages)

    def to_dict(self) -> Dict[str, Any]:
        """Convert thread to dictionary"""
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
        """Create thread from dictionary"""
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
