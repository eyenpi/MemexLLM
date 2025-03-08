"""Tests to improve coverage for SQLite storage implementation."""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from memexllm.core.models import Message, MessageRole, Thread
from memexllm.storage.sqlite import (
    DatabaseConnectionError,
    DatabaseIntegrityError,
    DatabaseOperationError,
    SQLiteStorage,
)
from memexllm.utils.exceptions import ThreadNotFoundError, ValidationError


@pytest.fixture
def db_path() -> Generator[str, None, None]:
    """Fixture to provide a test database path."""
    path = "test_sqlite_coverage.db"
    yield path
    # Cleanup after test
    if os.path.exists(path):
        os.remove(path)


def test_thread_to_row_edge_cases(db_path) -> None:
    """Test edge cases in _thread_to_row method."""
    storage = SQLiteStorage(db_path=db_path)

    # Test with None values in metadata
    thread = Thread(metadata={"null_value": None})
    row = storage._thread_to_row(thread)
    assert json.loads(row[3])["null_value"] is None

    # Test with empty metadata
    thread = Thread(metadata={})
    row = storage._thread_to_row(thread)
    assert json.loads(row[3]) == {}

    # Test with complex nested metadata
    complex_metadata = {
        "nested": {"level1": {"level2": "value"}},
        "array": [1, 2, 3],
        "mixed": [{"a": 1}, {"b": 2}],
    }
    thread = Thread(metadata=complex_metadata)
    row = storage._thread_to_row(thread)
    assert json.loads(row[3]) == complex_metadata


def test_get_thread_edge_cases(db_path) -> None:
    """Test edge cases in get_thread method."""
    storage = SQLiteStorage(db_path=db_path)

    # Test with thread that has no messages
    thread = Thread(metadata={"test": "data"})
    storage.save_thread(thread)

    retrieved_thread = storage.get_thread(thread.id)
    assert retrieved_thread is not None
    assert len(retrieved_thread.messages) == 0

    # Test with message_limit=None (should return all messages)
    retrieved_thread = storage.get_thread(thread.id, message_limit=None)
    assert retrieved_thread is not None

    # Test with non-existent thread
    assert storage.get_thread("nonexistent") is None

    # Test with message_limit=0 (should raise ValidationError)
    with pytest.raises(
        ValidationError, match="message_limit must be a positive integer"
    ):
        storage.get_thread(thread.id, message_limit=0)


def test_list_threads_edge_cases(db_path) -> None:
    """Test edge cases in list_threads method."""
    storage = SQLiteStorage(db_path=db_path)

    # Test with empty database
    threads = storage.list_threads()
    assert len(threads) == 0

    # Test with offset > number of threads
    for i in range(5):
        thread = Thread(metadata={"index": i})
        storage.save_thread(thread)

    threads = storage.list_threads(offset=10)
    assert len(threads) == 0

    # Test with limit=1
    threads = storage.list_threads(limit=1)
    assert len(threads) == 1


def test_delete_thread_edge_cases(db_path) -> None:
    """Test edge cases in delete_thread method."""
    storage = SQLiteStorage(db_path=db_path)

    # Test deleting non-existent thread
    assert storage.delete_thread("nonexistent") is False

    # Test deleting thread with messages
    thread = Thread(metadata={"test": "data"})
    storage.save_thread(thread)

    # Add a message to the thread
    thread.messages.append(Message(role="user", content="Hello"))
    storage.save_thread(thread)

    # Delete the thread
    assert storage.delete_thread(thread.id) is True

    # Verify thread is gone
    assert storage.get_thread(thread.id) is None


def test_search_threads_functionality(db_path) -> None:
    """Test search_threads functionality with various query types."""
    storage = SQLiteStorage(db_path=db_path)

    # Create threads with different metadata
    thread1 = Thread(metadata={"category": "work", "priority": "high"})
    thread2 = Thread(metadata={"category": "personal", "priority": "medium"})
    thread3 = Thread(metadata={"category": "work", "priority": "low"})

    storage.save_thread(thread1)
    storage.save_thread(thread2)
    storage.save_thread(thread3)

    # Test with metadata query
    results = storage.search_threads({"metadata": {"category": "work"}})
    assert len(results) == 2
    assert any(t.id == thread1.id for t in results)
    assert any(t.id == thread3.id for t in results)

    # Test with multiple conditions
    results = storage.search_threads(
        {"metadata": {"category": "work", "priority": "high"}}
    )
    assert len(results) == 1
    assert results[0].id == thread1.id

    # Test with no matches
    results = storage.search_threads({"metadata": {"category": "nonexistent"}})
    assert len(results) == 0


def test_search_threads_with_messages(db_path) -> None:
    """Test search_threads with message content."""
    storage = SQLiteStorage(db_path=db_path)

    # Create thread with messages
    thread = Thread()
    thread.messages.append(Message(role="user", content="Hello world"))
    thread.messages.append(Message(role="assistant", content="Hi there"))

    storage.save_thread(thread)

    # Test search by message content
    results = storage.search_threads({"content": "Hello"})
    assert len(results) == 1
    assert results[0].id == thread.id


def test_validation_errors(db_path) -> None:
    """Test validation errors in SQLiteStorage methods."""
    storage = SQLiteStorage(db_path=db_path)

    # Test empty thread_id in get_thread
    with pytest.raises(ValidationError, match="Thread ID cannot be empty"):
        storage.get_thread("")

    # Test negative limit in list_threads
    with pytest.raises(ValidationError, match="Limit must be a positive integer"):
        storage.list_threads(limit=0)

    # Test negative offset in list_threads
    with pytest.raises(ValidationError, match="Offset must be a non-negative integer"):
        storage.list_threads(offset=-1)

    # Test empty thread_id in delete_thread
    with pytest.raises(ValidationError, match="Thread ID cannot be empty"):
        storage.delete_thread("")

    # Test empty query in search_threads
    with pytest.raises(ValidationError, match="Search query cannot be empty"):
        storage.search_threads({})


def test_serialize_deserialize_metadata(db_path) -> None:
    """Test serialization and deserialization of metadata."""
    storage = SQLiteStorage(db_path=db_path)

    # Test serialization of complex metadata
    metadata = {
        "nested": {"level1": {"level2": "value"}},
        "array": [1, 2, 3],
        "mixed": [{"a": 1}, {"b": 2}],
        "null": None,
        "bool": True,
        "int": 42,
        "float": 3.14,
    }

    serialized = storage._serialize_metadata(metadata)
    deserialized = storage._deserialize_metadata(serialized)

    assert deserialized == metadata

    # Test deserialization of invalid JSON
    with pytest.raises(ValidationError, match="Failed to deserialize metadata"):
        storage._deserialize_metadata("{invalid json}")


def test_message_content_handling(db_path) -> None:
    """Test handling of message content."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a thread with a message that has empty string content
    thread = Thread()
    thread.messages.append(Message(role="user", content=""))
    storage.save_thread(thread)

    # Retrieve the thread and check the message content
    retrieved_thread = storage.get_thread(thread.id)
    assert retrieved_thread is not None
    assert len(retrieved_thread.messages) == 1
    assert (
        retrieved_thread.messages[0].content == ""
    )  # Empty string should be preserved


def test_message_to_row_implementation(db_path) -> None:
    """Test implementation details of _message_to_row method."""
    storage = SQLiteStorage(db_path=db_path)
    thread_id = "test_thread"

    # Test with empty string content
    message = Message(role="user", content="")
    row = storage._message_to_row(message, thread_id, 0)
    assert row[2] == ""  # Content should be empty string

    # Test with regular content
    message = Message(role="user", content="Hello")
    row = storage._message_to_row(message, thread_id, 0)
    assert row[2] == "Hello"

    # Test with None token_count
    message = Message(role="user", content="Hello", token_count=None)
    row = storage._message_to_row(message, thread_id, 0)
    assert row[6] is None  # token_count should be None

    # Test with integer token_count
    message = Message(role="user", content="Hello", token_count=10)
    row = storage._message_to_row(message, thread_id, 0)
    assert row[6] == 10


def test_sqlite_schema_coverage() -> None:
    """Test coverage for SQLiteSchema attributes."""
    from memexllm.storage.sqlite import SQLiteSchema

    # Access all schema attributes to ensure coverage
    assert SQLiteSchema.CREATE_THREADS_TABLE is not None
    assert SQLiteSchema.CREATE_MESSAGES_TABLE is not None
    assert SQLiteSchema.CREATE_MESSAGE_INDEX is not None
    assert SQLiteSchema.INSERT_THREAD is not None
    assert SQLiteSchema.INSERT_MESSAGE is not None
    assert SQLiteSchema.DELETE_THREAD_MESSAGES is not None
    assert SQLiteSchema.DELETE_THREAD is not None
    assert SQLiteSchema.GET_THREAD is not None
    assert SQLiteSchema.GET_THREAD_MESSAGES is not None
    assert SQLiteSchema.GET_ALL_THREAD_MESSAGES is not None
    assert SQLiteSchema.LIST_THREADS is not None
