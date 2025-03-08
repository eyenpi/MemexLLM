import logging
from functools import wraps
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
)

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionCallOptionParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCall,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

from ..algorithms.base import BaseAlgorithm
from ..core.history import HistoryManager
from ..core.models import (
    ImageContent,
    Message,
    MessageContent,
    MessageRole,
    TextContent,
    ToolCallContent,
)
from ..storage.base import BaseStorage
from ..utils.exceptions import (
    AuthenticationError,
    OpenAIAPIError,
    RateLimitError,
    ValidationError,
)

# Set up logging
logger = logging.getLogger(__name__)

# Type aliases for better readability
T = TypeVar("T", bound=Union[OpenAI, AsyncOpenAI])


def _convert_to_message(msg: Union[Dict[str, Any], ChatCompletionMessage]) -> Message:
    """
    Convert an OpenAI message format to internal Message format.

    Args:
        msg (Union[Dict[str, Any], ChatCompletionMessage]): Message in OpenAI format,
            either as a dictionary or ChatCompletionMessage object

    Returns:
        Message: Converted internal message format

    Raises:
        ValidationError: If the message format is invalid or contains invalid data
    """
    try:
        if isinstance(msg, dict):
            role = str(msg.get("role", ""))
            if role not in (
                "system",
                "user",
                "assistant",
                "tool",
                "function",
                "developer",
            ):
                raise ValidationError(
                    f"Invalid message role: '{role}'. Must be one of: system, user, assistant, tool, function, developer"
                )

            # Extract content
            content = msg.get("content", "")

            # Handle tool calls
            tool_calls = None
            if "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = [
                    ToolCallContent(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=tc.get("function", {}),
                    )
                    for tc in msg["tool_calls"]
                ]

            # Handle multimodal content
            if isinstance(content, list):
                processed_content: List[MessageContent] = []
                for item in content:
                    if item.get("type") == "text":
                        processed_content.append(TextContent(text=item.get("text", "")))
                    elif item.get("type") == "image_url":
                        processed_content.append(
                            ImageContent(
                                url=item.get("image_url", {}).get("url", ""),
                                detail=item.get("image_url", {}).get("detail"),
                            )
                        )
                    else:
                        logger.warning(f"Unknown content type: {item.get('type')}")
                content = processed_content

            return Message(
                role=cast(MessageRole, role),
                content=content,
                tool_calls=tool_calls,
                tool_call_id=msg.get("tool_call_id"),
                function_call=msg.get("function_call"),
                name=msg.get("name"),
            )
        else:
            # Handle ChatCompletionMessage object
            role = msg.role
            if role not in (
                "system",
                "user",
                "assistant",
                "tool",
                "function",
                "developer",
            ):
                raise ValidationError(
                    f"Invalid message role: '{role}'. Must be one of: system, user, assistant, tool, function, developer"
                )

            # Extract content
            content = msg.content

            # Handle tool calls
            tool_calls = None
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls = [
                    ToolCallContent(
                        id=tc.id,
                        type=tc.type,
                        function=cast(Dict[str, Any], tc.function),
                    )
                    for tc in msg.tool_calls
                ]

            return Message(
                role=cast(MessageRole, role),
                content=content,
                tool_calls=tool_calls,
                tool_call_id=getattr(msg, "tool_call_id", None),
                function_call=getattr(msg, "function_call", None),
                name=getattr(msg, "name", None),
            )
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        logger.error(f"Error converting OpenAI message format: {e}")
        raise ValidationError(f"Failed to convert OpenAI message format: {e}") from e


def _convert_to_openai_messages(
    messages: Sequence[Message],
) -> List[ChatCompletionMessageParam]:
    """
    Convert internal Message objects to OpenAI's message format.

    Args:
        messages (Sequence[Message]): List of internal Message objects to convert

    Returns:
        List[ChatCompletionMessageParam]: Messages formatted for OpenAI API
    """
    openai_messages: List[ChatCompletionMessageParam] = []

    for msg in messages:
        # Convert content to OpenAI format
        content = _convert_content_to_openai_format(msg.content)

        # Create message based on role
        if msg.role == "system":
            openai_messages.append(
                ChatCompletionSystemMessageParam(
                    role="system", content=cast(str, content)
                )
            )
        elif msg.role == "user":
            # For user messages, content can be string or list of content parts
            if isinstance(content, str):
                openai_messages.append(
                    ChatCompletionUserMessageParam(role="user", content=content)
                )
            else:
                # For multimodal content, we need to cast to the expected type
                content_parts = cast(
                    List[
                        Union[
                            ChatCompletionContentPartTextParam,
                            ChatCompletionContentPartImageParam,
                        ]
                    ],
                    content,
                )
                openai_messages.append(
                    ChatCompletionUserMessageParam(role="user", content=content_parts)
                )
        elif msg.role == "assistant":
            # For assistant messages, prepare parameters
            message_params: Dict[str, Any] = {"role": "assistant"}

            # Handle content (can be string or None)
            if content is not None:
                message_params["content"] = cast(Optional[str], content)

            # Add tool calls if present
            if msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    tool_calls.append(
                        {"id": tc.id, "type": tc.type, "function": tc.function}
                    )
                message_params["tool_calls"] = tool_calls

            # Add function call if present
            if msg.function_call:
                message_params["function_call"] = msg.function_call

            openai_messages.append(
                cast(ChatCompletionAssistantMessageParam, message_params)
            )
        elif msg.role == "tool":
            openai_messages.append(
                ChatCompletionToolMessageParam(
                    role="tool",
                    content=cast(str, content),
                    tool_call_id=msg.tool_call_id or "",
                )
            )
        elif msg.role == "function":
            openai_messages.append(
                ChatCompletionFunctionMessageParam(
                    role="function", content=cast(str, content), name=msg.name or ""
                )
            )
        elif msg.role == "developer":
            openai_messages.append(
                ChatCompletionDeveloperMessageParam(
                    role="developer", content=cast(str, content)
                )
            )

    return openai_messages


def _convert_content_to_openai_format(
    content: Union[str, List[MessageContent], None],
) -> Union[str, List[Dict[str, Any]], None]:
    """Convert content from internal format to OpenAI format.

    Args:
        content (Union[str, List[MessageContent], None]): Content in internal format

    Returns:
        Union[str, List[Dict[str, Any]], None]: Content in OpenAI format
    """
    if content is None:
        return None

    if isinstance(content, str):
        return content

    openai_content: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, TextContent):
            openai_content.append({"type": "text", "text": item.text})
        elif isinstance(item, ImageContent):
            image_url: Dict[str, Any] = {"url": item.url}
            if item.detail:
                image_url["detail"] = item.detail
            openai_content.append(
                {
                    "type": "image_url",
                    "image_url": image_url,
                }
            )

    return openai_content


def with_history(
    storage: Optional[BaseStorage] = None,
    algorithm: Optional[BaseAlgorithm] = None,
    history_manager: Optional[HistoryManager] = None,
) -> Callable[[T], T]:
    """
    Decorator to add conversation history management to an OpenAI client.

    This decorator adds a thread_id parameter to the chat.completions.create method,
    which enables automatic conversation history management. When a thread_id is
    provided, messages are stored and retrieved from the specified storage backend.

    Args:
        storage (Optional[BaseStorage]): Storage backend for conversation history.
            Required if history_manager is not provided.
        algorithm (Optional[BaseAlgorithm]): Algorithm for managing conversation history.
            If None, all messages are included in the context.
        history_manager (Optional[HistoryManager]): Pre-configured history manager.
            If provided, storage and algorithm parameters are ignored.

    Returns:
        Callable[[T], T]: Decorator function that adds history management to an OpenAI client

    Raises:
        ValidationError: If neither storage nor history_manager is provided

    Example:
        ```python
        # Create a client with history management
        storage = MemoryStorage()
        client = with_history(storage=storage)(OpenAI())

        # Use the client with a thread_id
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            thread_id="thread_123"
        )
        ```
    """
    from ..utils.exceptions import ConfigurationError

    if not storage and not history_manager:
        raise ConfigurationError(
            "Either storage or history_manager must be provided to with_history decorator"
        )

    # Create history manager if not provided
    manager = history_manager or HistoryManager(
        storage=cast(BaseStorage, storage), algorithm=algorithm
    )

    def decorator(client: T) -> T:
        nonlocal manager

        # Store original methods
        original_chat_completions_create = client.chat.completions.create

        def _prepare_messages(
            thread_id: str,
            new_messages: Sequence[Union[Dict[str, Any], ChatCompletionMessage]],
        ) -> List[ChatCompletionMessageParam]:
            """
            Prepare messages by combining thread history with new messages.

            This function:
            1. Retrieves existing thread history
            2. Handles system message overrides
            3. Combines history with new messages
            4. Converts all messages to OpenAI format
            5. Automatically includes assistant's message with tool calls when a tool response is present
            6. Ensures assistant messages with tool calls are immediately followed by their tool responses

            Args:
                thread_id (str): ID of the conversation thread
                new_messages (Sequence[Union[Dict[str, Any], ChatCompletionMessage]]):
                    New messages to add to the conversation

            Returns:
                List[ChatCompletionMessageParam]: Combined and formatted messages
            """
            thread = manager.get_thread(thread_id)
            converted_messages = [_convert_to_message(msg) for msg in new_messages]

            if not thread:
                return _convert_to_openai_messages(converted_messages)

            # Extract system message if present in new messages
            system_message = next(
                (msg for msg in converted_messages if msg.role == "system"),
                None,
            )

            # Check if there's a tool response in the new messages
            tool_responses = [msg for msg in converted_messages if msg.role == "tool"]

            # Prepare thread messages
            thread_messages: List[Message] = []

            # Initialize assistant_tool_pairs
            assistant_tool_pairs = []

            # If we have tool responses, we need to include the corresponding assistant messages with tool calls
            # and ensure they're in the correct order
            if tool_responses:
                # For each tool response, find the corresponding assistant message with the tool call
                for tool_response in tool_responses:
                    tool_call_id = tool_response.tool_call_id

                    if tool_call_id:
                        # Find the assistant message with this tool call ID
                        assistant_with_tool_call = None
                        for msg in thread.messages:
                            if msg.role == "assistant" and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    # Compare the tool call ID with the ID in the tool response
                                    if tc.id == tool_call_id:
                                        assistant_with_tool_call = msg
                                        break
                                if assistant_with_tool_call:
                                    break

                        # If found, add it to our list of messages to include
                        if assistant_with_tool_call:
                            # Store the assistant message and tool response as a pair
                            assistant_tool_pairs.append(
                                (assistant_with_tool_call, tool_response)
                            )

                # Add other messages from thread history (excluding assistant messages with tool calls and their responses)
                for msg in thread.messages:
                    # Skip system message from history if we have a new one
                    if msg.role == "system" and system_message:
                        continue

                    # Skip assistant messages with tool calls (they'll be added in the correct order later)
                    if msg.role == "assistant" and msg.tool_calls:
                        if any(msg == pair[0] for pair in assistant_tool_pairs):
                            continue

                    # Skip tool responses that we're handling (they'll be added after their assistant messages)
                    if msg.role == "tool" and msg.tool_call_id:
                        if any(
                            msg.tool_call_id == pair[1].tool_call_id
                            for pair in assistant_tool_pairs
                        ):
                            continue

                    thread_messages.append(msg)
            else:
                # If no tool responses, just add all messages from thread history
                for msg in thread.messages:
                    # Skip system message from history if we have a new one
                    if msg.role == "system" and system_message:
                        continue
                    thread_messages.append(msg)

            # Combine messages
            if system_message:
                thread_messages.insert(0, system_message)

            # Add new messages (excluding system message if it was handled)
            for msg in converted_messages:
                if (
                    system_message
                    and msg.role == "system"
                    and msg.content == system_message.content
                ):
                    continue

                # Skip tool responses that we're handling (they'll be added after their assistant messages)
                if msg.role == "tool" and msg.tool_call_id:
                    if any(
                        msg.tool_call_id == pair[1].tool_call_id
                        for pair in assistant_tool_pairs
                    ):
                        continue

                thread_messages.append(msg)

            # Now add the assistant-tool pairs in the correct order
            for assistant_msg, tool_msg in assistant_tool_pairs:
                # Find the right position to insert the pair
                # We want to insert them at the end, unless there are messages that should come after them
                thread_messages.append(assistant_msg)
                thread_messages.append(tool_msg)

            return _convert_to_openai_messages(thread_messages)

        @wraps(original_chat_completions_create)
        async def async_chat_completions_create(
            *args: Any, thread_id: Optional[str] = None, **kwargs: Any
        ) -> ChatCompletion:
            """
            Async version of chat completions with history management.

            Args:
                thread_id (Optional[str]): ID of the conversation thread.
                    If not provided, a new thread will be created.
                *args: Arguments passed to the original create method
                **kwargs: Keyword arguments passed to the original create method

            Returns:
                ChatCompletion: The API response from OpenAI

            Raises:
                TypeError: If the API response is not a ChatCompletion
            """
            # Create or get thread
            if not thread_id:
                thread = manager.create_thread()
                thread_id = thread.id

            # Get messages and prepare them with history
            new_messages = kwargs.get("messages", [])
            prepared_messages = _prepare_messages(thread_id, new_messages)
            kwargs["messages"] = prepared_messages

            # Call original method
            response = await original_chat_completions_create(*args, **kwargs)
            if not isinstance(response, ChatCompletion):
                raise TypeError("Expected ChatCompletion response")

            # Add new messages and response to history
            for msg in new_messages:
                converted_msg = _convert_to_message(msg)
                # Convert complex content to string for storage if needed
                content_for_storage = _prepare_content_for_storage(
                    converted_msg.content
                )
                manager.add_message(
                    thread_id=thread_id,
                    content=content_for_storage,
                    role=converted_msg.role,
                    metadata={"type": "input"},
                    tool_calls=converted_msg.tool_calls,
                    tool_call_id=converted_msg.tool_call_id,
                    function_call=converted_msg.function_call,
                    name=converted_msg.name,
                )

            if isinstance(response, ChatCompletion):
                for choice in response.choices:
                    if isinstance(choice.message, ChatCompletionMessage):
                        converted_msg = _convert_to_message(choice.message)
                        # Convert complex content to string for storage if needed
                        content_for_storage = _prepare_content_for_storage(
                            converted_msg.content
                        )
                        manager.add_message(
                            thread_id=thread_id,
                            content=content_for_storage,
                            role=converted_msg.role,
                            metadata={
                                "type": "output",
                                "finish_reason": choice.finish_reason,
                                "model": response.model,
                            },
                            tool_calls=converted_msg.tool_calls,
                            tool_call_id=converted_msg.tool_call_id,
                            function_call=converted_msg.function_call,
                            name=converted_msg.name,
                        )

            return response

        @wraps(original_chat_completions_create)
        def sync_chat_completions_create(
            *args: Any, thread_id: Optional[str] = None, **kwargs: Any
        ) -> ChatCompletion:
            """
            Sync version of chat completions with history management.

            Args:
                thread_id (Optional[str]): ID of the conversation thread.
                    If not provided, a new thread will be created.
                *args: Arguments passed to the original create method
                **kwargs: Keyword arguments passed to the original create method

            Returns:
                ChatCompletion: The API response from OpenAI

            Raises:
                TypeError: If the API response is not a ChatCompletion
            """
            # Create or get thread
            if not thread_id:
                thread = manager.create_thread()
                thread_id = thread.id

            # Get messages and prepare them with history
            new_messages = kwargs.get("messages", [])
            prepared_messages = _prepare_messages(thread_id, new_messages)
            kwargs["messages"] = prepared_messages

            # Call original method
            response = original_chat_completions_create(*args, **kwargs)
            if not isinstance(response, ChatCompletion):
                raise TypeError("Expected ChatCompletion response")

            # Add new messages and response to history
            for msg in new_messages:
                converted_msg = _convert_to_message(msg)
                # Convert complex content to string for storage if needed
                content_for_storage = _prepare_content_for_storage(
                    converted_msg.content
                )
                manager.add_message(
                    thread_id=thread_id,
                    content=content_for_storage,
                    role=converted_msg.role,
                    metadata={"type": "input"},
                    tool_calls=converted_msg.tool_calls,
                    tool_call_id=converted_msg.tool_call_id,
                    function_call=converted_msg.function_call,
                    name=converted_msg.name,
                )

            if isinstance(response, ChatCompletion):
                for choice in response.choices:
                    if isinstance(choice.message, ChatCompletionMessage):
                        converted_msg = _convert_to_message(choice.message)
                        # Convert complex content to string for storage if needed
                        content_for_storage = _prepare_content_for_storage(
                            converted_msg.content
                        )
                        manager.add_message(
                            thread_id=thread_id,
                            content=content_for_storage,
                            role=converted_msg.role,
                            metadata={
                                "type": "output",
                                "finish_reason": choice.finish_reason,
                                "model": response.model,
                            },
                            tool_calls=converted_msg.tool_calls,
                            tool_call_id=converted_msg.tool_call_id,
                            function_call=converted_msg.function_call,
                            name=converted_msg.name,
                        )

            return response

        def _prepare_content_for_storage(
            content: Union[str, List[MessageContent], None],
        ) -> str:
            """Convert content to a string representation for storage.

            Args:
                content (Union[str, List[MessageContent], None]): Content to convert

            Returns:
                str: String representation of the content
            """
            if content is None:
                return ""

            if isinstance(content, str):
                return content

            # For structured content, create a simple text representation
            text_parts = []
            for item in content:
                if isinstance(item, TextContent):
                    text_parts.append(item.text)
                elif isinstance(item, ImageContent):
                    text_parts.append(f"[Image: {item.url}]")
                elif isinstance(item, ToolCallContent):
                    text_parts.append(f"[Tool Call: {item.type} - {item.id}]")

            return " ".join(text_parts) if text_parts else ""

        # Replace methods with wrapped versions
        if isinstance(client, AsyncOpenAI):
            client.chat.completions.create = async_chat_completions_create  # type: ignore
        else:
            client.chat.completions.create = sync_chat_completions_create  # type: ignore

        return client

    return decorator
