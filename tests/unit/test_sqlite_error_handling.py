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
    with pytest.raises(ValueError, match="Database path cannot be empty"):
        SQLiteStorage(db_path="")

    # Test negative max_messages
    with pytest.raises(ValueError, match="max_messages must be a positive integer"):
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
    with pytest.raises(ValueError, match="Thread cannot be None"):
        storage.save_thread(None)

    # Test thread with empty ID
    thread = Thread(id="")
    with pytest.raises(ValueError, match="Thread ID cannot be empty"):
        storage.save_thread(thread)


@patch.object(SQLiteStorage, "_get_connection")
def test_save_thread_database_error(mock_get_connection, db_path):
    """Test database error handling in save_thread."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a mock connection that raises an error
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.Error("Database error")
    mock_get_connection.return_value = mock_conn

    thread = Thread(id="test-thread")

    with pytest.raises(
        DatabaseOperationError, match="Database error while saving thread"
    ):
        storage.save_thread(thread)

    # Verify rollback was called
    mock_conn.rollback.assert_called_once()


@patch.object(SQLiteStorage, "_get_connection")
def test_save_thread_integrity_error(mock_get_connection, db_path):
    """Test integrity error handling in save_thread."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a mock connection that raises an integrity error
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = sqlite3.IntegrityError("Integrity error")
    mock_get_connection.return_value = mock_conn

    thread = Thread(id="test-thread")

    with pytest.raises(
        DatabaseIntegrityError, match="Database integrity error while saving thread"
    ):
        storage.save_thread(thread)

    # Verify rollback was called
    mock_conn.rollback.assert_called_once()


def test_get_thread_validation(db_path):
    """Test validation in get_thread method."""
    storage = SQLiteStorage(db_path=db_path)

    # Test empty thread_id
    with pytest.raises(ValueError, match="Thread ID cannot be empty"):
        storage.get_thread("")

    # Test negative message_limit
    with pytest.raises(ValueError, match="message_limit must be a positive integer"):
        storage.get_thread("test-id", message_limit=-1)


@patch.object(SQLiteStorage, "_get_connection")
def test_get_thread_database_error(mock_get_connection, db_path):
    """Test database error handling in get_thread."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a mock connection that raises an error
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.side_effect = sqlite3.Error("Database error")
    mock_get_connection.return_value = mock_conn

    with pytest.raises(
        DatabaseOperationError, match="Database error while retrieving thread"
    ):
        storage.get_thread("test-id")


def test_list_threads_validation(db_path):
    """Test validation in list_threads method."""
    storage = SQLiteStorage(db_path=db_path)

    # Test non-positive limit
    with pytest.raises(ValueError, match="Limit must be a positive integer"):
        storage.list_threads(limit=0)

    # Test negative offset
    with pytest.raises(ValueError, match="Offset must be a non-negative integer"):
        storage.list_threads(offset=-1)


@patch.object(SQLiteStorage, "_get_connection")
def test_list_threads_database_error(mock_get_connection, db_path):
    """Test database error handling in list_threads."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a mock connection that raises an error
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
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
    with pytest.raises(ValueError, match="Thread ID cannot be empty"):
        storage.delete_thread("")


@patch.object(SQLiteStorage, "_get_connection")
def test_delete_thread_database_error(mock_get_connection, db_path):
    """Test database error handling in delete_thread."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a mock connection that raises an error
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.execute.side_effect = sqlite3.Error("Database error")
    mock_get_connection.return_value = mock_conn

    with pytest.raises(
        DatabaseOperationError, match="Database error while deleting thread"
    ):
        storage.delete_thread("test-id")


def test_search_threads_validation(db_path):
    """Test validation in search_threads method."""
    storage = SQLiteStorage(db_path=db_path)

    # Test empty query
    with pytest.raises(ValueError, match="Search query cannot be empty"):
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
        storage.search_threads({"content": "test"})


def test_serialize_metadata_error(db_path):
    """Test error handling in _serialize_metadata."""
    storage = SQLiteStorage(db_path=db_path)

    # Create a metadata object that can't be serialized
    class UnserializableObject:
        pass

    metadata = {"object": UnserializableObject()}

    with pytest.raises(ValueError, match="Failed to serialize metadata"):
        storage._serialize_metadata(metadata)


def test_deserialize_metadata_error(db_path):
    """Test error handling in _deserialize_metadata."""
    storage = SQLiteStorage(db_path=db_path)

    # Invalid JSON string
    invalid_json = "{invalid json"

    with pytest.raises(ValueError, match="Failed to deserialize metadata"):
        storage._deserialize_metadata(invalid_json)


def test_thread_to_row_validation(db_path):
    """Test validation in _thread_to_row."""
    storage = SQLiteStorage(db_path=db_path)

    # Thread with empty ID
    thread = Thread(id="")

    with pytest.raises(ValueError, match="Thread ID cannot be empty"):
        storage._thread_to_row(thread)


def test_message_to_row_validation(db_path):
    """Test validation in _message_to_row."""
    storage = SQLiteStorage(db_path=db_path)

    # Message with empty ID
    message = Message(id="", content="test", role="user")

    with pytest.raises(ValueError, match="Message ID cannot be empty"):
        storage._message_to_row(message, "thread-id", 0)

    # Empty thread_id
    message = Message(id="msg-id", content="test", role="user")

    with pytest.raises(ValueError, match="Thread ID cannot be empty"):
        storage._message_to_row(message, "", 0)

    # Negative index
    with pytest.raises(ValueError, match="Message index must be non-negative"):
        storage._message_to_row(message, "thread-id", -1)


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
    with pytest.raises(ValueError):
        storage.save_thread(thread2)

    # Verify we can still retrieve the valid thread
    retrieved_thread = storage.get_thread("thread-1")
    assert retrieved_thread is not None
    assert retrieved_thread.id == "thread-1"
    assert len(retrieved_thread.messages) == 1
    assert retrieved_thread.messages[0].id == "msg-1"
