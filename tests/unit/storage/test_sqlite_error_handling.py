"""Tests for SQLite storage error handling."""

import os
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from memexllm.core.models import Message, Thread
from memexllm.storage.sqlite import (
    DatabaseConnectionError,
    DatabaseIntegrityError,
    DatabaseOperationError,
    SQLiteStorage,
)
from memexllm.utils.exceptions import ValidationError


@pytest.fixture
def db_path():
    """Fixture to provide a temporary database path."""
    path = "test_error_handling.db"
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_init_with_invalid_parameters():
    """Test initialization with invalid parameters."""
    # Test empty db_path
    with pytest.raises(ValidationError, match="Database path cannot be empty"):
        SQLiteStorage(db_path="")

    # Test negative max_messages
    with pytest.raises(
        ValidationError, match="max_messages must be a positive integer"
    ):
        SQLiteStorage(max_messages=-1)


@patch("sqlite3.connect")
def test_connection_error(mock_connect, db_path):
    """Test handling of connection errors."""
    # Simulate connection error
    mock_connect.side_effect = sqlite3.Error("Connection failed")

    with pytest.raises(DatabaseConnectionError, match="Failed to connect to database"):
        SQLiteStorage(db_path=db_path)


def test_save_thread_validation(db_path):
    """Test validation in save_thread method."""
    storage = SQLiteStorage(db_path=db_path)

    # Test None thread
    with pytest.raises(ValidationError, match="Thread cannot be None"):
        storage.save_thread(None)

    # Test thread with empty ID
    thread = Thread(id="")
    with pytest.raises(ValidationError, match="Thread ID cannot be empty"):
        storage.save_thread(thread)


@patch.object(SQLiteStorage, "_get_connection")
def test_save_thread_database_error(mock_get_connection, db_path):
    """Test database error handling in save_thread."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a mock connection that raises an error
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.Error("Database error")
    mock_get_connection.return_value = mock_conn

    # Create a valid thread
    thread = Thread(id="test-thread")
    thread.messages = [Message(id="msg-1", content="Hello", role="user")]

    with pytest.raises(
        DatabaseOperationError, match="Database error while saving thread"
    ):
        storage.save_thread(thread)


@patch.object(SQLiteStorage, "_get_connection")
def test_save_thread_integrity_error(mock_get_connection, db_path):
    """Test integrity error handling in save_thread."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a mock connection that raises an integrity error
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.IntegrityError("Integrity error")
    mock_get_connection.return_value = mock_conn

    # Create a valid thread
    thread = Thread(id="test-thread")
    thread.messages = [Message(id="msg-1", content="Hello", role="user")]

    with pytest.raises(
        DatabaseIntegrityError, match="Database integrity error while saving thread"
    ):
        storage.save_thread(thread)


def test_get_thread_validation(db_path):
    """Test validation in get_thread method."""
    storage = SQLiteStorage(db_path=db_path)

    # Test empty thread_id
    with pytest.raises(ValidationError, match="Thread ID cannot be empty"):
        storage.get_thread("")

    # Test negative message_limit
    with pytest.raises(
        ValidationError, match="message_limit must be a positive integer"
    ):
        storage.get_thread("test", message_limit=-1)


@patch.object(SQLiteStorage, "_get_connection")
def test_get_thread_database_error(mock_get_connection, db_path):
    """Test database error handling in get_thread."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a mock connection that raises an error
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.Error("Database error")
    mock_get_connection.return_value = mock_conn

    with pytest.raises(
        DatabaseOperationError, match="Database error while retrieving thread"
    ):
        storage.get_thread("test-thread")


def test_list_threads_validation(db_path):
    """Test validation in list_threads method."""
    storage = SQLiteStorage(db_path=db_path)

    # Test negative limit
    with pytest.raises(ValidationError, match="Limit must be a positive integer"):
        storage.list_threads(limit=-1)

    # Test negative offset
    with pytest.raises(ValidationError, match="Offset must be a non-negative integer"):
        storage.list_threads(offset=-1)


@patch.object(SQLiteStorage, "_get_connection")
def test_list_threads_database_error(mock_get_connection, db_path):
    """Test database error handling in list_threads."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a mock connection that raises an error
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.Error("Database error")
    mock_get_connection.return_value = mock_conn

    with pytest.raises(
        DatabaseOperationError, match="Database error while listing threads"
    ):
        storage.list_threads()


def test_delete_thread_validation(db_path):
    """Test validation in delete_thread method."""
    storage = SQLiteStorage(db_path=db_path)

    # Test empty thread_id
    with pytest.raises(ValidationError, match="Thread ID cannot be empty"):
        storage.delete_thread("")


@patch.object(SQLiteStorage, "_get_connection")
def test_delete_thread_database_error(mock_get_connection, db_path):
    """Test database error handling in delete_thread."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a mock connection that raises an error
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.Error("Database error")
    mock_get_connection.return_value = mock_conn

    with pytest.raises(
        DatabaseOperationError, match="Database error while deleting thread"
    ):
        storage.delete_thread("test-thread")


def test_search_threads_validation(db_path):
    """Test validation in search_threads method."""
    storage = SQLiteStorage(db_path=db_path)

    # Test empty query
    with pytest.raises(ValidationError, match="Search query cannot be empty"):
        storage.search_threads({})


@patch.object(SQLiteStorage, "_get_connection")
def test_search_threads_database_error(mock_get_connection, db_path):
    """Test database error handling in search_threads."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a mock connection that raises an error
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.Error("Database error")
    mock_get_connection.return_value = mock_conn

    with pytest.raises(
        DatabaseOperationError, match="Database error during thread search"
    ):
        storage.search_threads({"metadata": {"key": "value"}})


def test_serialize_metadata_error(db_path):
    """Test error handling in _serialize_metadata."""
    storage = SQLiteStorage(db_path=db_path)

    class UnserializableObject:
        """Object that can't be serialized to JSON."""

        pass

    with pytest.raises(ValidationError, match="Failed to serialize metadata"):
        storage._serialize_metadata({"obj": UnserializableObject()})


def test_deserialize_metadata_error(db_path):
    """Test error handling in _deserialize_metadata."""
    storage = SQLiteStorage(db_path=db_path)

    with pytest.raises(ValidationError, match="Failed to deserialize metadata"):
        storage._deserialize_metadata("{invalid json")


def test_thread_to_row_validation(db_path):
    """Test validation in _thread_to_row."""
    storage = SQLiteStorage(db_path=db_path)

    # Test thread with empty ID
    thread = Thread(id="")
    with pytest.raises(ValidationError, match="Thread ID cannot be empty"):
        storage._thread_to_row(thread)


def test_message_to_row_validation(db_path):
    """Test validation in _message_to_row."""
    storage = SQLiteStorage(db_path=db_path)

    # Test message with empty ID
    msg = Message(id="", content="Hello", role="user")
    with pytest.raises(ValidationError, match="Message ID cannot be empty"):
        storage._message_to_row(msg, "thread-id", 0)

    # Test empty thread_id
    msg = Message(id="msg-id", content="Hello", role="user")
    with pytest.raises(ValidationError, match="Thread ID cannot be empty"):
        storage._message_to_row(msg, "", 0)

    # Test negative index
    with pytest.raises(ValidationError, match="Message index must be non-negative"):
        storage._message_to_row(msg, "thread-id", -1)


def test_integration_error_recovery(db_path):
    """Test that the system can recover from errors."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a valid thread
    thread1 = Thread(id="thread-1")
    thread1.messages = [Message(id="msg-1", content="Hello", role="user")]

    # Save the thread
    storage.save_thread(thread1)

    # Try to save an invalid thread (should raise an error)
    thread2 = Thread(id="")
    with pytest.raises(ValidationError, match="Thread ID cannot be empty"):
        storage.save_thread(thread2)

    # Verify the first thread is still accessible
    retrieved_thread = storage.get_thread("thread-1")
    assert retrieved_thread is not None
    assert retrieved_thread.id == "thread-1"
    assert len(retrieved_thread.messages) == 1
    assert retrieved_thread.messages[0].content == "Hello"
