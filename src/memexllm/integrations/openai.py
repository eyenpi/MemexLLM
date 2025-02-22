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
    """Convert external message to internal message format."""
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
    """Convert internal Message objects to OpenAI message format."""
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
    Decorator that wraps an OpenAI client to add history management capabilities.

    Args:
        storage: Storage backend implementation (optional if history_manager is provided)
        algorithm: History management algorithm (optional if history_manager is provided)
        history_manager: HistoryManager instance (optional, will be created if not provided)

    Returns:
        A wrapped OpenAI client that automatically records chat history
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
            """Prepare messages by combining thread history with new messages."""
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
