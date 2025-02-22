from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from memexllm.algorithms.base import BaseAlgorithm
from memexllm.core.models import Message, Thread
from memexllm.integrations.openai import with_history
from memexllm.storage.base import BaseStorage


class MockStorage(BaseStorage):
    def __init__(self) -> None:
        self.threads: dict[str, Thread] = {}

    def save_thread(self, thread: Thread) -> None:
        self.threads[thread.id] = thread

    def get_thread(self, thread_id: str) -> Thread | None:
        return self.threads.get(thread_id)

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

    def create_thread(self) -> Thread:
        thread = Thread()
        self.save_thread(thread)
        return thread


class MockAlgorithm(BaseAlgorithm):
    def process_thread(self, thread: Thread, new_message: Message) -> None:
        thread.add_message(new_message)


@pytest.fixture
def storage() -> MockStorage:
    return MockStorage()


@pytest.fixture
def algorithm() -> MockAlgorithm:
    return MockAlgorithm()


@pytest.fixture
def mock_chat_completion() -> ChatCompletion:
    return ChatCompletion(
        id="test-completion",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="Test response", role="assistant"
                ),
            )
        ],
        created=1234567890,
        model="gpt-4",
        object="chat.completion",
        usage={"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
    )


def test_sync_client_wrapper(
    storage: MockStorage, algorithm: MockAlgorithm, mock_chat_completion: ChatCompletion
) -> None:
    # Create mock client with proper structure
    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_chat_completion)

    # Wrap client
    wrapped_client = with_history(storage=storage, algorithm=algorithm)(mock_client)

    # Make request
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    response = wrapped_client.chat.completions.create(messages=messages)

    # Verify response
    assert response == mock_chat_completion

    # Verify history
    threads = storage.list_threads()
    assert len(threads) == 1
    thread: Thread = threads[0]

    # Verify messages were saved
    assert len(thread.messages) == 3  # 2 input messages + 1 response
    assert thread.messages[0].role == "user"
    assert thread.messages[0].content == "Hello"
    assert thread.messages[1].role == "assistant"
    assert thread.messages[1].content == "Hi there"
    assert thread.messages[2].role == "assistant"
    assert thread.messages[2].content == "Test response"


@pytest.mark.asyncio
async def test_async_client_wrapper(
    storage: MockStorage, algorithm: MockAlgorithm, mock_chat_completion: ChatCompletion
) -> None:
    # Create mock client with proper structure
    mock_client = MagicMock(spec=AsyncOpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_chat_completion)

    # Wrap client
    wrapped_client = with_history(storage=storage, algorithm=algorithm)(mock_client)

    # Make request
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]

    response = await wrapped_client.chat.completions.create(messages=messages)

    # Verify response
    assert response == mock_chat_completion

    # Verify history
    threads = storage.list_threads()
    assert len(threads) == 1
    thread: Thread = threads[0]

    # Verify messages were saved
    assert len(thread.messages) == 3  # 2 input messages + 1 response
    assert thread.messages[0].role == "user"
    assert thread.messages[0].content == "Hello"
    assert thread.messages[1].role == "assistant"
    assert thread.messages[1].content == "Hi there"
    assert thread.messages[2].role == "assistant"
    assert thread.messages[2].content == "Test response"


def test_thread_reuse(
    storage: MockStorage, algorithm: MockAlgorithm, mock_chat_completion: ChatCompletion
) -> None:
    # Create mock client with proper structure
    mock_client = MagicMock(spec=OpenAI)
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    mock_client.chat.completions.create = MagicMock(return_value=mock_chat_completion)

    # Wrap client
    wrapped_client = with_history(storage=storage, algorithm=algorithm)(mock_client)

    # Create thread
    initial_thread = storage.create_thread()
    thread_id = initial_thread.id

    # Make first request
    messages1 = [{"role": "user", "content": "Hello"}]
    response1 = wrapped_client.chat.completions.create(
        messages=messages1, thread_id=thread_id
    )

    # Make second request
    messages2 = [{"role": "user", "content": "How are you?"}]
    response2 = wrapped_client.chat.completions.create(
        messages=messages2, thread_id=thread_id
    )

    # Verify responses
    assert response1 == mock_chat_completion
    assert response2 == mock_chat_completion

    # Verify history
    maybe_thread = storage.get_thread(thread_id)
    assert maybe_thread is not None, "Thread should exist"
    thread: Thread = maybe_thread

    # Verify all messages were saved in the same thread
    assert len(thread.messages) == 4  # 2 user messages + 2 responses
    assert thread.messages[0].role == "user"
    assert thread.messages[0].content == "Hello"
    assert thread.messages[1].role == "assistant"
    assert thread.messages[1].content == "Test response"
    assert thread.messages[2].role == "user"
    assert thread.messages[2].content == "How are you?"
    assert thread.messages[3].role == "assistant"
    assert thread.messages[3].content == "Test response"


@pytest.mark.integration
def test_real_openai_integration(
    storage: MockStorage, algorithm: MockAlgorithm
) -> None:
    # Note: This test requires a valid OPENAI_API_KEY environment variable
    client = OpenAI()
    wrapped_client = with_history(storage=storage, algorithm=algorithm)(client)

    # Make request
    messages = [{"role": "user", "content": "Say 'test' and nothing else"}]
    response = wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, max_tokens=10
    )

    # Verify response and history
    threads = storage.list_threads()
    assert len(threads) == 1
    thread: Thread = threads[0]

    # Verify messages were saved
    assert len(thread.messages) == 2  # 1 input message + 1 response
    assert thread.messages[0].role == "user"
    assert thread.messages[0].content == "Say 'test' and nothing else"
    assert thread.messages[1].role == "assistant"
    assert "test" in thread.messages[1].content.lower()
