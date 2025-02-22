from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from ..algorithms.base import BaseAlgorithm
from ..core.history import HistoryManager
from ..core.models import MessageRole
from ..storage.base import BaseStorage


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
            thread_id: str, new_messages: List[Dict[str, str]]
        ) -> List[Dict[str, str]]:
            """Prepare messages by combining thread history with new messages."""
            thread = history_manager.get_thread(thread_id)
            if not thread:
                return new_messages

            # Extract system message if present in new messages
            system_message = next(
                (msg for msg in new_messages if msg["role"] == "system"), None
            )

            # Get system message from history if not in new messages
            if not system_message:
                system_message = next(
                    (
                        {"role": msg.role, "content": msg.content}
                        for msg in thread.messages
                        if msg.role == "system"
                    ),
                    None,
                )

            # Start with system message if present
            final_messages = [system_message] if system_message else []

            # Add history messages from thread
            history_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in thread.messages
                if msg.role
                != "system"  # Skip system messages from history as we handle them separately
            ]

            # Add new non-system messages
            new_user_messages = [msg for msg in new_messages if msg["role"] != "system"]

            # Combine messages in order: system (if any) -> history -> new messages
            final_messages.extend(history_messages)

            # Only add new messages that aren't already in history
            seen_messages = {(msg["role"], msg["content"]) for msg in final_messages}
            for msg in new_user_messages:
                if (msg["role"], msg["content"]) not in seen_messages:
                    final_messages.append(msg)

            # Remove None entries
            final_messages = [msg for msg in final_messages if msg is not None]

            return final_messages

        @wraps(original_chat_completions_create)
        async def async_chat_completions_create(
            *args: Any, **kwargs: Any
        ) -> ChatCompletion:
            # Create or get thread from metadata
            thread_id = kwargs.pop("thread_id", None)  # Remove thread_id from kwargs
            if not thread_id:
                thread = history_manager.create_thread()
                thread_id = thread.id

            # Get messages and prepare them with history
            new_messages = kwargs.get("messages", [])
            prepared_messages = _prepare_messages(thread_id, new_messages)
            kwargs["messages"] = prepared_messages

            # Call original method
            response = await original_chat_completions_create(*args, **kwargs)

            # Add new messages and response to history
            for msg in new_messages:
                history_manager.add_message(
                    thread_id=thread_id,
                    content=msg["content"],
                    role=msg["role"],
                    metadata={"type": "input"},
                )

            if isinstance(response, ChatCompletion):
                for choice in response.choices:
                    if isinstance(choice.message, ChatCompletionMessage):
                        history_manager.add_message(
                            thread_id=thread_id,
                            content=choice.message.content or "",
                            role=choice.message.role,
                            metadata={
                                "type": "output",
                                "finish_reason": choice.finish_reason,
                                "model": response.model,
                            },
                        )

            return response

        @wraps(original_chat_completions_create)
        def sync_chat_completions_create(*args: Any, **kwargs: Any) -> ChatCompletion:
            # Create or get thread from metadata
            thread_id = kwargs.pop("thread_id", None)  # Remove thread_id from kwargs
            if not thread_id:
                thread = history_manager.create_thread()
                thread_id = thread.id

            # Get messages and prepare them with history
            new_messages = kwargs.get("messages", [])
            prepared_messages = _prepare_messages(thread_id, new_messages)
            kwargs["messages"] = prepared_messages

            # Call original method
            response = original_chat_completions_create(*args, **kwargs)

            # Add new messages and response to history
            for msg in new_messages:
                history_manager.add_message(
                    thread_id=thread_id,
                    content=msg["content"],
                    role=msg["role"],
                    metadata={"type": "input"},
                )

            if isinstance(response, ChatCompletion):
                for choice in response.choices:
                    if isinstance(choice.message, ChatCompletionMessage):
                        history_manager.add_message(
                            thread_id=thread_id,
                            content=choice.message.content or "",
                            role=choice.message.role,
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
