from typing import Any, Dict, List, cast
from unittest.mock import MagicMock, patch

import pytest

from memexllm.algorithms import BaseAlgorithm, FIFOAlgorithm
from memexllm.core import HistoryManager, Message, MessageRole, Thread
from memexllm.storage import BaseStorage, MemoryStorage
from memexllm.utils.exceptions import StorageError, ThreadNotFoundError, ValidationError


def test_history_manager_validation_errors():
    """Test validation errors in HistoryManager methods"""
    # Test None storage
    with pytest.raises(ValidationError, match="Storage cannot be None"):
        HistoryManager(storage=None)  # type: ignore

    # Setup for other tests
    storage = MemoryStorage()
    manager = HistoryManager(storage=storage)

    # Test empty thread_id in get_thread
    with pytest.raises(ValidationError, match="Thread ID cannot be empty"):
        manager.get_thread("")

    # Test empty thread_id in add_message
    with pytest.raises(ValidationError, match="Thread ID cannot be empty"):
        manager.add_message(
            thread_id="",
            content="Hello",
            role=cast(MessageRole, "user"),
        )

    # Test invalid content type in add_message
    with pytest.raises(ValidationError, match="Content must be a string or None"):
        manager.add_message(
            thread_id="thread_id",
            content=123,  # type: ignore
            role=cast(MessageRole, "user"),
        )

    # Test invalid role in add_message
    with pytest.raises(ValidationError, match="Invalid role"):
        manager.add_message(
            thread_id="thread_id",
            content="Hello",
            role=cast(MessageRole, "invalid_role"),
        )

    # Test empty thread_id in get_messages
    with pytest.raises(ValidationError, match="Thread ID cannot be empty"):
        manager.get_messages("")

    # Test invalid limit in list_threads
    with pytest.raises(ValidationError, match="Limit must be a positive integer"):
        manager.list_threads(limit=0)

    # Test invalid offset in list_threads
    with pytest.raises(ValidationError, match="Offset must be a non-negative integer"):
        manager.list_threads(offset=-1)

    # Test empty thread_id in delete_thread
    with pytest.raises(ValidationError, match="Thread ID cannot be empty"):
        manager.delete_thread("")


def test_algorithm_integration():
    """Test integration with history management algorithms"""
    storage = MemoryStorage()
    algorithm = FIFOAlgorithm(max_messages=2)
    manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create thread and add messages
    thread = manager.create_thread()

    # Add 3 messages (exceeding the algorithm's max_messages)
    manager.add_message(
        thread_id=thread.id, content="Message 1", role=cast(MessageRole, "user")
    )
    manager.add_message(
        thread_id=thread.id, content="Message 2", role=cast(MessageRole, "assistant")
    )
    manager.add_message(
        thread_id=thread.id, content="Message 3", role=cast(MessageRole, "user")
    )

    # Get thread - should only have the last 2 messages due to FIFO algorithm
    updated_thread = manager.get_thread(thread.id)
    assert updated_thread is not None
    assert len(updated_thread.messages) == 2
    assert updated_thread.messages[0].content == "Message 2"
    assert updated_thread.messages[1].content == "Message 3"

    # Verify all messages are still in storage
    all_messages = manager.get_messages(thread.id)
    assert len(all_messages) == 3


def test_storage_error_handling():
    """Test error handling for storage errors"""
    # Create a mock storage that raises exceptions
    mock_storage = MagicMock(spec=BaseStorage)
    mock_storage.save_thread.side_effect = Exception("Storage error")
    mock_storage.get_thread.side_effect = Exception("Storage error")
    mock_storage.list_threads.side_effect = Exception("Storage error")
    mock_storage.delete_thread.side_effect = Exception("Storage error")

    manager = HistoryManager(storage=mock_storage)

    # Test create_thread error handling
    with pytest.raises(StorageError, match="Failed to create thread"):
        manager.create_thread()

    # Test get_thread error handling
    with pytest.raises(StorageError, match="Failed to get thread"):
        manager.get_thread("thread_id")

    # Test add_message error handling
    mock_storage.get_thread.side_effect = None
    mock_storage.get_thread.return_value = Thread(id="thread_id")
    with pytest.raises(StorageError, match="Failed to add message to thread"):
        manager.add_message(
            thread_id="thread_id",
            content="Hello",
            role=cast(MessageRole, "user"),
        )

    # Test get_messages error handling
    mock_storage.get_thread.side_effect = Exception("Storage error")
    with pytest.raises(StorageError, match="Failed to get messages"):
        manager.get_messages("thread_id")

    # Test list_threads error handling
    with pytest.raises(StorageError, match="Failed to list threads"):
        manager.list_threads()

    # Test delete_thread error handling
    with pytest.raises(StorageError, match="Failed to delete thread"):
        manager.delete_thread("thread_id")


def test_add_message_with_all_parameters():
    """Test add_message with all optional parameters"""
    storage = MemoryStorage()
    manager = HistoryManager(storage=storage)

    # Create thread
    thread = manager.create_thread()

    # Add message with all parameters
    metadata = {"source": "test"}
    tool_calls = [{"type": "function", "function": {"name": "test_function"}}]
    function_call = {"name": "test_function", "arguments": "{}"}

    message = manager.add_message(
        thread_id=thread.id,
        content="Hello",
        role=cast(MessageRole, "assistant"),
        metadata=metadata,
        tool_calls=tool_calls,
        tool_call_id="tool_123",
        function_call=function_call,
        name="assistant_name",
    )

    # Verify message was created with all parameters
    assert message.content == "Hello"
    assert message.role == "assistant"
    assert message.metadata == metadata
    assert message.tool_calls == tool_calls
    assert message.tool_call_id == "tool_123"
    assert message.function_call == function_call
    assert message.name == "assistant_name"

    # Verify message was added to thread
    updated_thread = manager.get_thread(thread.id)
    assert updated_thread is not None
    assert len(updated_thread.messages) == 1
    assert updated_thread.messages[0].id == message.id


def test_thread_not_found_handling():
    """Test handling of ThreadNotFoundError in various methods"""
    storage = MemoryStorage()
    manager = HistoryManager(storage=storage)

    # Test add_message with non-existent thread
    with pytest.raises(ThreadNotFoundError, match="Thread not found: nonexistent"):
        manager.add_message(
            thread_id="nonexistent",
            content="Hello",
            role=cast(MessageRole, "user"),
        )

    # Test get_messages with non-existent thread
    with pytest.raises(ThreadNotFoundError, match="Thread not found: nonexistent"):
        manager.get_messages("nonexistent")


def test_delete_thread_edge_cases():
    """Test edge cases for delete_thread method"""
    # Test with mock storage that returns False
    mock_storage = MagicMock(spec=BaseStorage)
    mock_storage.delete_thread.return_value = False

    manager = HistoryManager(storage=mock_storage)
    assert manager.delete_thread("thread_id") is False

    # Test with mock storage that returns True
    mock_storage.delete_thread.return_value = True
    assert manager.delete_thread("thread_id") is True


def test_add_message_validation_error_propagation():
    """Test that ValidationError is propagated in add_message"""
    storage = MemoryStorage()
    manager = HistoryManager(storage=storage)

    # Create a thread
    thread = manager.create_thread()

    # Mock the storage to raise ValidationError when saving thread
    mock_storage = MagicMock(spec=BaseStorage)
    mock_storage.get_thread.return_value = Thread(id=thread.id)
    mock_storage.save_thread.side_effect = ValidationError("Validation error")

    manager.storage = mock_storage

    # Test that ValidationError is propagated
    with pytest.raises(ValidationError, match="Validation error"):
        manager.add_message(
            thread_id=thread.id,
            content="Hello",
            role=cast(MessageRole, "user"),
        )
