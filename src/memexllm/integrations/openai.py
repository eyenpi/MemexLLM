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
        ValueError: If the message role is invalid
    """
    if isinstance(msg, dict):
        role = str(msg.get("role", ""))
        if role not in ("system", "user", "assistant", "tool", "function", "developer"):
            raise ValueError(f"Invalid role: {role}")

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
        role = str(msg.role)
        if role not in ("system", "user", "assistant", "tool", "function", "developer"):
            raise ValueError(f"Invalid role: {role}")

        # Extract content
        content = msg.content or ""

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

        # Handle multimodal content (if content is a list of content parts)
        if isinstance(content, list):
            processed_content_obj: List[MessageContent] = []
            for item in content:
                if hasattr(item, "type"):
                    if item.type == "text":
                        processed_content_obj.append(TextContent(text=item.text))
                    elif item.type == "image_url":
                        processed_content_obj.append(
                            ImageContent(
                                url=item.image_url.url,
                                detail=getattr(item.image_url, "detail", None),
                            )
                        )
            content = processed_content_obj

        return Message(
            role=cast(MessageRole, role),
            content=content,
            tool_calls=tool_calls,
            tool_call_id=getattr(msg, "tool_call_id", None),
            function_call=getattr(msg, "function_call", None),
            name=getattr(msg, "name", None),
        )


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
    Decorator that adds conversation history management to an OpenAI client.

    This decorator wraps an OpenAI client to automatically track and manage conversation
    history. It supports both synchronous and asynchronous clients and handles thread
    creation, message storage, and history management.

    Tool calls are handled automatically. When you send a tool response message, the
    history manager will automatically include the corresponding assistant message with
    the tool call in the request to OpenAI. This means you only need to send the tool
    response message, and the history manager will take care of the rest.

    The decorator ensures that assistant messages with tool calls are immediately followed
    by their tool response messages, as required by the OpenAI API. This is handled
    transparently, so you don't need to worry about the order of messages.

    Example with tool calls:
    ```python
    # First request with tools
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What's the weather?"}],
        tools=[...],
        thread_id="my-thread"
    )

    # Get the tool call information
    tool_call = response.choices[0].message.tool_calls[0]

    # Process the tool call and get the result
    tool_result = my_weather_function(tool_call.function.arguments)

    # Send only the tool response - the history manager will automatically include
    # the assistant's message with the tool call
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "tool",
                "content": json.dumps(tool_result),
                "tool_call_id": tool_call.id  # Use the actual tool call ID from the assistant's response
            }
        ],
        thread_id="my-thread"
    )
    ```

    Args:
        storage (Optional[BaseStorage]): Storage backend for persisting conversation history.
            Required if history_manager is not provided.
        algorithm (Optional[BaseAlgorithm]): Algorithm for managing conversation history.
            Optional, used for features like context window management.
        history_manager (Optional[HistoryManager]): Existing HistoryManager instance.
            If provided, storage and algorithm parameters are ignored.

    Returns:
        Callable: A decorator function that wraps an OpenAI client

    Raises:
        ValueError: If neither history_manager nor storage is provided

    Example:
        ```python
        from openai import OpenAI
        from memexllm.storage import SQLiteStorage
        from memexllm.algorithms import FIFOAlgorithm

        # Create client with history management
        client = OpenAI()
        storage = SQLiteStorage("chat_history.db")
        algorithm = FIFOAlgorithm(max_messages=50)

        client = with_history(storage=storage, algorithm=algorithm)(client)

        # Use client with automatic history tracking
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            thread_id="my-thread"  # Optional, will be created if not provided
        )
        ```
    """

    def decorator(client: T) -> T:
        nonlocal history_manager
        if history_manager is None:
            if storage is None:
                raise ValueError("Either history_manager or storage must be provided")
            history_manager = HistoryManager(storage=storage, algorithm=algorithm)

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
            thread = history_manager.get_thread(thread_id)
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
                thread = history_manager.create_thread()
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
                history_manager.add_message(
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
                        history_manager.add_message(
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
                thread = history_manager.create_thread()
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
                history_manager.add_message(
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
                        history_manager.add_message(
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
