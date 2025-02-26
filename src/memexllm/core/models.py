import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union, cast

MessageRole = Literal["user", "assistant", "system", "tool", "function", "developer"]


@dataclass
class MessageContent:
    """
    Base class for different types of message content.
    """

    pass


@dataclass
class TextContent(MessageContent):
    """
    Represents text content in a message.

    Attributes:
        text (str): The text content
    """

    text: str


@dataclass
class ImageContent(MessageContent):
    """
    Represents image content in a message.

    Attributes:
        url (str): URL of the image
        detail (Optional[str]): Detail level of the image (e.g., "auto", "low", "high")
    """

    url: str
    detail: Optional[str] = None


@dataclass
class ToolCallContent(MessageContent):
    """
    Represents a tool call in a message.

    Attributes:
        id (str): Unique identifier for the tool call
        type (str): Type of the tool call (e.g., "function")
        function (Dict[str, Any]): Function details including name and arguments
    """

    id: str
    type: str
    function: Dict[str, Any]


@dataclass
class Message:
    """
    Represents a single message in a conversation thread.

    A message contains the content, the role of the sender, and associated metadata.

    Attributes:
        content (Union[str, List[MessageContent]]): The content of the message, either as a string or structured content
        role (MessageRole): Role of the message sender ("user", "assistant", "system", etc.)
        id (str): Unique identifier for the message (UUID)
        created_at (datetime): UTC timestamp when the message was created
        metadata (Dict[str, Any]): Additional custom metadata for the message
        token_count (Optional[int]): Number of tokens in the message content, if calculated
        tool_calls (Optional[List[ToolCallContent]]): Tool calls made in this message
        tool_call_id (Optional[str]): ID of the tool call this message is responding to
        function_call (Optional[Dict[str, Any]]): Function call details if this message contains a function call
        name (Optional[str]): Name field for function messages
    """

    content: Union[str, List[MessageContent]]
    role: MessageRole
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None
    tool_calls: Optional[List[ToolCallContent]] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """
        Create a Message instance from a dictionary representation.

        Args:
            data (Dict[str, Any]): Dictionary containing message data with keys:
                - content: Message content (string or structured content)
                - role: Message sender role
                - id (optional): Message unique identifier
                - metadata (optional): Additional message metadata
                - token_count (optional): Number of tokens in content
                - tool_calls (optional): List of tool calls
                - tool_call_id (optional): ID of the tool call this message responds to
                - function_call (optional): Function call details
                - name (optional): Name field for function messages

        Returns:
            Message: A new Message instance
        """
        content = data["content"]

        # Process structured content if it's a list
        if isinstance(content, list):
            processed_content: List[MessageContent] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        processed_content.append(TextContent(text=item.get("text", "")))
                    elif item.get("type") == "image":
                        processed_content.append(
                            ImageContent(
                                url=item.get("image_url", {}).get("url", ""),
                                detail=item.get("image_url", {}).get("detail"),
                            )
                        )
            content = processed_content if processed_content else content

        # Process tool calls
        tool_calls = None
        if "tool_calls" in data:
            tool_calls = [
                ToolCallContent(
                    id=tc.get("id", ""),
                    type=tc.get("type", ""),
                    function=tc.get("function", {}),
                )
                for tc in data["tool_calls"]
            ]

        return cls(
            content=content,
            role=cast(MessageRole, data["role"]),
            id=data.get("id", str(uuid.uuid4())),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id"),
            function_call=data.get("function_call"),
            name=data.get("name"),
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
                    "content": self._serialize_content(msg.content),
                    "created_at": msg.created_at.isoformat(),
                    "metadata": msg.metadata,
                    "token_count": msg.token_count,
                    **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
                    **({"tool_call_id": msg.tool_call_id} if msg.tool_call_id else {}),
                    **(
                        {"function_call": msg.function_call}
                        if msg.function_call
                        else {}
                    ),
                    **({"name": msg.name} if msg.name else {}),
                }
                for msg in self.messages
            ],
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def _serialize_content(
        self, content: Union[str, List[MessageContent]]
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Serialize message content to a format suitable for JSON serialization.

        Args:
            content (Union[str, List[MessageContent]]): The message content to serialize

        Returns:
            Union[str, List[Dict[str, Any]]]: Serialized content
        """
        if isinstance(content, str):
            return content

        serialized: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, TextContent):
                serialized.append({"type": "text", "text": item.text})
            elif isinstance(item, ImageContent):
                image_url: Dict[str, Any] = {"url": item.url}
                if item.detail:
                    image_url["detail"] = item.detail
                serialized.append(
                    {
                        "type": "image",
                        "image_url": image_url,
                    }
                )
            elif isinstance(item, ToolCallContent):
                serialized.append(
                    {"id": item.id, "type": item.type, "function": item.function}
                )

        return serialized

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
