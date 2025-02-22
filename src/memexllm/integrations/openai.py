from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

from ..algorithms.base import BaseAlgorithm
from ..core.history import HistoryManager
from ..core.models import Message, MessageRole
from ..storage.base import BaseStorage


def _convert_to_message(msg: Union[Dict[str, Any], ChatCompletionMessage]) -> Message:
    """
    Convert an OpenAI message format to internal Message format.

    Args:
        msg (Union[Dict[str, Any], ChatCompletionMessage]): Message in OpenAI format,
            either as a dictionary or ChatCompletionMessage object

    Returns:
        Message: Converted internal message format

    Raises:
        ValueError: If the message role is invalid (must be system/user/assistant)
    """
    if isinstance(msg, dict):
        role = str(msg.get("role", ""))
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Invalid role: {role}")
        return Message(
            role=cast(MessageRole, role),
            content=str(msg.get("content", "")),
        )
    else:
        role = str(msg.role)
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Invalid role: {role}")
        return Message(
            role=cast(MessageRole, role),
            content=msg.content or "",
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
        if msg.role == "system":
            openai_messages.append(
                ChatCompletionSystemMessageParam(role="system", content=msg.content)
            )
        elif msg.role == "user":
            openai_messages.append(
                ChatCompletionUserMessageParam(role="user", content=msg.content)
            )
        elif msg.role == "assistant":
            openai_messages.append(
                ChatCompletionAssistantMessageParam(
                    role="assistant", content=msg.content
                )
            )

    return openai_messages


def with_history(
    storage: Optional[BaseStorage] = None,
    algorithm: Optional[BaseAlgorithm] = None,
    history_manager: Optional[HistoryManager] = None,
) -> Callable[[Union[OpenAI, AsyncOpenAI]], Union[OpenAI, AsyncOpenAI]]:
    """
    Decorator that adds conversation history management to an OpenAI client.

    This decorator wraps an OpenAI client to automatically track and manage conversation
    history. It supports both synchronous and asynchronous clients and handles thread
    creation, message storage, and history management.

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

    def decorator(client: Union[OpenAI, AsyncOpenAI]) -> Union[OpenAI, AsyncOpenAI]:
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

            # Prepare thread messages
            thread_messages: List[Message] = []
            for msg in thread.messages:
                if msg.role == "system" and system_message:
                    continue  # Skip system message from history if we have a new one
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
                thread_messages.append(msg)

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
                history_manager.add_message(
                    thread_id=thread_id,
                    content=converted_msg.content,
                    role=converted_msg.role,
                    metadata={"type": "input"},
                )

            if isinstance(response, ChatCompletion):
                for choice in response.choices:
                    if isinstance(choice.message, ChatCompletionMessage):
                        converted_msg = _convert_to_message(choice.message)
                        history_manager.add_message(
                            thread_id=thread_id,
                            content=converted_msg.content,
                            role=converted_msg.role,
                            metadata={
                                "type": "output",
                                "finish_reason": choice.finish_reason,
                                "model": response.model,
                            },
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
                history_manager.add_message(
                    thread_id=thread_id,
                    content=converted_msg.content,
                    role=converted_msg.role,
                    metadata={"type": "input"},
                )

            if isinstance(response, ChatCompletion):
                for choice in response.choices:
                    if isinstance(choice.message, ChatCompletionMessage):
                        converted_msg = _convert_to_message(choice.message)
                        history_manager.add_message(
                            thread_id=thread_id,
                            content=converted_msg.content,
                            role=converted_msg.role,
                            metadata={
                                "type": "output",
                                "finish_reason": choice.finish_reason,
                                "model": response.model,
                            },
                        )

            return response

        # Replace methods with wrapped versions
        if isinstance(client, AsyncOpenAI):
            client.chat.completions.create = async_chat_completions_create  # type: ignore
        else:
            client.chat.completions.create = sync_chat_completions_create  # type: ignore

        return client

    return decorator
