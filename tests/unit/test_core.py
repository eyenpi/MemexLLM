import pytest

from memexllm.algorithms.fifo import FIFOAlgorithm
from memexllm.core.history import HistoryManager
from memexllm.core.models import Message, Thread
from memexllm.storage.memory import MemoryStorage


def test_history_manager_creation() -> None:
    storage = MemoryStorage()
    algorithm = FIFOAlgorithm(max_messages=50)
    manager = HistoryManager(storage=storage, algorithm=algorithm)
    assert manager.storage == storage
    assert manager.algorithm == algorithm


def test_thread_creation() -> None:
    manager = HistoryManager(storage=MemoryStorage())
    metadata = {"user_id": "test123"}
    thread = manager.create_thread(metadata=metadata)

    assert thread.id is not None
    assert thread.metadata == metadata
    assert len(thread.messages) == 0


def test_message_management() -> None:
    manager = HistoryManager(storage=MemoryStorage())
    thread = manager.create_thread()

    # Add messages
    manager.add_message(thread_id=thread.id, content="Hello", role="user")
    manager.add_message(thread_id=thread.id, content="Hi there!", role="assistant")

    # Retrieve and verify
    updated_thread = manager.get_thread(thread.id)
    assert updated_thread is not None  # Type check
    assert len(updated_thread.messages) == 2
    assert updated_thread.messages[0].content == "Hello"
    assert updated_thread.messages[0].role == "user"
    assert updated_thread.messages[1].content == "Hi there!"
    assert updated_thread.messages[1].role == "assistant"


def test_thread_model_methods() -> None:
    """Test Thread model methods that were previously uncovered"""
    from datetime import datetime, timezone

    # Test to_dict and from_dict with full data
    thread = Thread(
        id="test-id",
        metadata={"key": "value"},
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    message = Message(
        id="msg-id",
        content="Hello",
        role="user",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        metadata={"msg_key": "msg_value"},
        token_count=10,
    )
    thread.add_message(message)

    # Test to_dict
    thread_dict = thread.to_dict()
    assert thread_dict["id"] == "test-id"
    assert thread_dict["metadata"] == {"key": "value"}
    assert thread_dict["created_at"] == "2024-01-01T00:00:00+00:00"
    # Don't test exact updated_at since it's set automatically
    assert "updated_at" in thread_dict

    # Test message serialization
    msg_dict = thread_dict["messages"][0]
    assert msg_dict["id"] == "msg-id"
    assert msg_dict["content"] == "Hello"
    assert msg_dict["role"] == "user"
    assert msg_dict["metadata"] == {"msg_key": "msg_value"}
    assert msg_dict["token_count"] == 10

    # Test from_dict
    new_thread = Thread.from_dict(thread_dict)
    assert new_thread.id == thread.id
    assert new_thread.metadata == thread.metadata
    assert new_thread.created_at == thread.created_at
    # Don't compare updated_at since it changes
    assert len(new_thread.messages) == 1
    assert new_thread.messages[0].id == message.id
    assert new_thread.messages[0].token_count == message.token_count


def test_thread_property_methods() -> None:
    """Test Thread property methods"""
    thread = Thread()

    # Test message_count property
    assert thread.message_count == 0

    thread.add_message(Message(content="Hello", role="user"))
    assert thread.message_count == 1

    # Test get_messages method
    messages = thread.get_messages()
    assert len(messages) == 1
    assert messages[0].content == "Hello"
