from typing import Any, Dict, List, Optional, Union, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from memexllm.algorithms import BaseAlgorithm
from memexllm.core import HistoryManager, Message, MessageRole, Thread
from memexllm.integrations.openai import _convert_to_openai_messages, with_history
from memexllm.storage import BaseStorage


class MockStorage(BaseStorage):
    def __init__(self) -> None:
        super().__init__()
        self.threads: dict[str, Thread] = {}

    def save_thread(self, thread: Thread) -> None:
        self.threads[thread.id] = thread

    def get_thread(
        self, thread_id: str, message_limit: Optional[int] = None
    ) -> Thread | None:
        thread = self.threads.get(thread_id)
        if not thread:
            return None

        if message_limit is not None and len(thread.messages) > message_limit:
            thread_copy = Thread(id=thread.id, metadata=thread.metadata)
            thread_copy.messages = thread.messages[-message_limit:]
            return thread_copy
        return thread

    def list_threads(self, limit: int = 100, offset: int = 0) -> list[Thread]:
        return list(self.threads.values())[offset : offset + limit]

    def delete_thread(self, thread_id: str) -> bool:
        if thread_id in self.threads:
            del self.threads[thread_id]
            return True
        return False

    def search_threads(self, query: dict[str, Any]) -> list[Thread]:
        # Simple mock implementation that returns all threads
        return list(self.threads.values())


class MockAlgorithm(BaseAlgorithm):
    def process_thread(self, thread: Thread, new_message: Message) -> None:
        thread.add_message(new_message)

    def get_message_window(self, messages: List[Message]) -> List[Message]:
        # Simple implementation that returns all messages
        return messages


@pytest.fixture
def storage() -> MockStorage:
    return MockStorage()


@pytest.fixture
def algorithm() -> MockAlgorithm:
    return MockAlgorithm()


@pytest.fixture
def mock_chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-123",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="Hello there!",
                    role="assistant",
                ),
            )
        ],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage=CompletionUsage(total_tokens=10, prompt_tokens=5, completion_tokens=5),
    )


@pytest.fixture
def history_manager(storage: MockStorage, algorithm: MockAlgorithm) -> HistoryManager:
    return HistoryManager(storage=storage, algorithm=algorithm)


def test_sync_client_wrapper(
    history_manager: HistoryManager, mock_chat_completion: ChatCompletion
) -> None:
    # Create mock client with proper structure
    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_chat_completion)

    # Wrap client
    wrapped_client = with_history(history_manager=history_manager)(mock_client)

    # Create a new thread using history manager
    thread = history_manager.create_thread()

    # Create messages and convert to OpenAI format
    messages = [
        Message(role=cast(MessageRole, "user"), content="Hello"),
        Message(role=cast(MessageRole, "assistant"), content="Hi there"),
    ]
    openai_messages = _convert_to_openai_messages(messages)

    response = wrapped_client.chat.completions.create(  # type: ignore
        messages=openai_messages, model="gpt-4", thread_id=thread.id
    )

    # Verify response
    assert response == mock_chat_completion

    # Verify history was recorded
    retrieved_thread = history_manager.get_thread(thread.id)
    assert retrieved_thread is not None
    thread = retrieved_thread
    assert len(thread.messages) == 3  # 2 input messages + 1 response
    assert thread.messages[0].role == "user"
    assert thread.messages[0].content == "Hello"
    assert thread.messages[1].role == "assistant"
    assert thread.messages[1].content == "Hi there"
    assert thread.messages[2].role == "assistant"
    assert thread.messages[2].content == "Hello there!"


@pytest.mark.asyncio
async def test_async_client_wrapper(
    history_manager: HistoryManager, mock_chat_completion: ChatCompletion
) -> None:
    # Create mock client with proper structure
    mock_client = MagicMock(spec=AsyncOpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)

    # Wrap client
    wrapped_client = with_history(history_manager=history_manager)(mock_client)

    # Create a new thread using history manager
    thread = history_manager.create_thread()

    # Create messages and convert to OpenAI format
    messages = [
        Message(role=cast(MessageRole, "user"), content="Hello"),
        Message(role=cast(MessageRole, "assistant"), content="Hi there"),
    ]
    openai_messages = _convert_to_openai_messages(messages)

    response = await wrapped_client.chat.completions.create(  # type: ignore
        messages=openai_messages, model="gpt-4", thread_id=thread.id
    )

    # Verify response
    assert response == mock_chat_completion

    # Verify history was recorded
    retrieved_thread = history_manager.get_thread(thread.id)
    assert retrieved_thread is not None
    thread = retrieved_thread
    assert len(thread.messages) == 3
    assert thread.messages[0].role == "user"
    assert thread.messages[0].content == "Hello"
    assert thread.messages[1].role == "assistant"
    assert thread.messages[1].content == "Hi there"
    assert thread.messages[2].role == "assistant"
    assert thread.messages[2].content == "Hello there!"


def test_thread_reuse(
    history_manager: HistoryManager, mock_chat_completion: ChatCompletion
) -> None:
    # Create mock client with proper structure
    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_chat_completion)

    # Wrap client
    wrapped_client = with_history(history_manager=history_manager)(mock_client)

    # Create a new thread using history manager
    thread = history_manager.create_thread()

    # Make first request using Message objects and convert to OpenAI format
    messages1 = [Message(role=cast(MessageRole, "user"), content="Hello")]
    openai_messages1 = _convert_to_openai_messages(messages1)

    response1 = wrapped_client.chat.completions.create(  # type: ignore
        messages=openai_messages1, model="gpt-4", thread_id=thread.id
    )

    # Make second request using Message objects and convert to OpenAI format
    messages2 = [Message(role=cast(MessageRole, "user"), content="How are you?")]
    openai_messages2 = _convert_to_openai_messages(messages2)

    response2 = wrapped_client.chat.completions.create(  # type: ignore
        messages=openai_messages2, model="gpt-4", thread_id=thread.id
    )

    # Verify responses
    assert response1 == mock_chat_completion
    assert response2 == mock_chat_completion

    # Verify history was recorded correctly
    retrieved_thread = history_manager.get_thread(thread.id)
    assert retrieved_thread is not None
    thread = retrieved_thread
    assert len(thread.messages) == 4
    assert thread.messages[0].role == "user"
    assert thread.messages[0].content == "Hello"
    assert thread.messages[1].role == "assistant"
    assert thread.messages[1].content == "Hello there!"
    assert thread.messages[2].role == "user"
    assert thread.messages[2].content == "How are you?"
    assert thread.messages[3].role == "assistant"
    assert thread.messages[3].content == "Hello there!"
