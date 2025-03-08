"""SQLite storage backend for MemexLLM."""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiosqlite

from ..core.models import Message, Thread
from ..utils.exceptions import (
    ConnectionError,
    IntegrityError,
    MessageNotFoundError,
    OperationError,
    ResourceNotFoundError,
    ThreadNotFoundError,
    ValidationError,
)
from .base import BaseStorage

# Set up logging
logger = logging.getLogger(__name__)


class SQLiteStorageError(Exception):
    """Base exception for SQLite storage errors."""

    pass


class DatabaseConnectionError(SQLiteStorageError):
    """Exception raised when database connection fails."""

    pass


class DatabaseOperationError(SQLiteStorageError):
    """Exception raised when a database operation fails."""

    pass


class DatabaseIntegrityError(SQLiteStorageError):
    """Exception raised when a database integrity constraint is violated."""

    pass


class SQLiteSchema:
    """SQL schemas and queries for the SQLite storage backend."""

    CREATE_THREADS_TABLE = """
        CREATE TABLE IF NOT EXISTS threads (
            id TEXT PRIMARY KEY,
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL,
            metadata TEXT NOT NULL
        )
    """

    CREATE_MESSAGES_TABLE = """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            content TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            metadata TEXT NOT NULL,
            token_count INTEGER,
            message_index INTEGER NOT NULL,
            FOREIGN KEY (thread_id) REFERENCES threads (id) ON DELETE CASCADE
        )
    """

    CREATE_MESSAGE_INDEX = """
        CREATE INDEX IF NOT EXISTS idx_messages_thread_id 
        ON messages (thread_id, message_index)
    """

    INSERT_THREAD = """
        INSERT OR REPLACE INTO threads 
        (id, created_at, updated_at, metadata)
        VALUES (?, ?, ?, ?)
    """

    INSERT_MESSAGE = """
        INSERT INTO messages 
        (id, thread_id, content, role, created_at, metadata, token_count, message_index)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    DELETE_THREAD_MESSAGES = "DELETE FROM messages WHERE thread_id = ?"
    DELETE_THREAD = "DELETE FROM threads WHERE id = ?"
    GET_THREAD = "SELECT * FROM threads WHERE id = ?"
    GET_THREAD_MESSAGES = """
        SELECT * FROM messages 
        WHERE thread_id = ? 
        ORDER BY message_index DESC
        LIMIT ?
    """
    GET_ALL_THREAD_MESSAGES = """
        SELECT * FROM messages 
        WHERE thread_id = ? 
        ORDER BY message_index DESC
    """
    LIST_THREADS = """
        SELECT * FROM threads 
        ORDER BY updated_at DESC 
        LIMIT ? OFFSET ?
    """


class SQLiteStorage(BaseStorage):
    """
    SQLite storage backend for conversation threads.

    This storage backend persists threads and messages to a SQLite database file,
    making it suitable for production use with moderate data volumes.

    Attributes:
        db_path (str): Path to the SQLite database file
        max_messages (Optional[int]): Maximum number of messages to store per thread
    """

    def __init__(
        self, db_path: str = "memexllm.db", max_messages: Optional[int] = None
    ):
        """
        Initialize SQLite storage with database path.

        Args:
            db_path (str): Path to the SQLite database file. If the file doesn't
                exist, it will be created. Defaults to "memexllm.db" in the
                current directory.
            max_messages (Optional[int]): Maximum number of messages to store per thread.
                If None, store all messages.

        Raises:
            ConnectionError: If the database connection cannot be established
        """
        # Validate input parameters
        if not db_path:
            raise ValidationError("Database path cannot be empty")

        if max_messages is not None and max_messages <= 0:
            raise ValidationError("max_messages must be a positive integer")

        self.db_path = db_path or ":memory:"
        self.max_messages = max_messages
        logger.debug(f"Initializing SQLite storage with db_path: {self.db_path}")

        # Create database directory if it doesn't exist
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        # Initialize database
        try:
            conn = self._get_connection()
            self._init_db(conn)
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseConnectionError(f"Failed to connect to database: {e}") from e

    def _init_db(self, conn: sqlite3.Connection) -> None:
        """Initialize database schema.

        Raises:
            DatabaseConnectionError: If the database cannot be connected to
            DatabaseOperationError: If schema creation fails
        """
        try:
            conn.execute(SQLiteSchema.CREATE_THREADS_TABLE)
            conn.execute(SQLiteSchema.CREATE_MESSAGES_TABLE)
            conn.execute(SQLiteSchema.CREATE_MESSAGE_INDEX)
            conn.commit()
            logger.debug(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database schema: {e}")
            raise DatabaseOperationError(
                f"Error initializing database schema: {e}"
            ) from e

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper configuration.

        Returns:
            A configured SQLite connection

        Raises:
            DatabaseConnectionError: If the connection cannot be established
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            # Enable foreign key support
            conn.execute("PRAGMA foreign_keys = ON")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise DatabaseConnectionError(f"Failed to connect to database: {e}") from e

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> str:
        """Serialize metadata to JSON string.

        Args:
            metadata: Dictionary of metadata

        Returns:
            JSON string representation

        Raises:
            ValidationError: If metadata cannot be serialized
        """
        try:
            return json.dumps(metadata)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize metadata: {e}")
            raise ValidationError(f"Failed to serialize metadata: {e}") from e

    def _deserialize_metadata(self, metadata_str: str) -> Dict[str, Any]:
        """Deserialize metadata from JSON string.

        Args:
            metadata_str: JSON string of metadata

        Returns:
            Dictionary of metadata

        Raises:
            ValidationError: If metadata cannot be deserialized
        """
        try:
            result: Dict[str, Any] = json.loads(metadata_str)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize metadata: {e}")
            raise ValidationError(f"Failed to deserialize metadata: {e}") from e

    def _thread_to_row(self, thread: Thread) -> Tuple:
        """Convert Thread object to database row values.

        Args:
            thread: Thread to convert

        Returns:
            Tuple of values for database insertion

        Raises:
            ValidationError: If thread data is invalid
        """
        if not thread.id:
            raise ValidationError("Thread ID cannot be empty")

        try:
            return (
                thread.id,
                thread.created_at.isoformat(),
                thread.updated_at.isoformat(),
                self._serialize_metadata(thread.metadata),
            )
        except Exception as e:
            logger.error(f"Failed to convert thread to row: {e}")
            raise ValidationError(f"Failed to convert thread to row: {e}") from e

    def _message_to_row(self, msg: Message, thread_id: str, index: int) -> Tuple:
        """Convert Message object to database row values.

        Args:
            msg: Message to convert
            thread_id: ID of the parent thread
            index: Position of message in thread

        Returns:
            Tuple of values for database insertion

        Raises:
            ValidationError: If message data is invalid
        """
        if not msg.id:
            raise ValidationError("Message ID cannot be empty")

        if not thread_id:
            raise ValidationError("Thread ID cannot be empty")

        if index < 0:
            raise ValidationError("Message index must be non-negative")

        try:
            return (
                msg.id,
                thread_id,
                msg.content,
                msg.role,
                msg.created_at.isoformat(),
                self._serialize_metadata(msg.metadata),
                msg.token_count,
                index,
            )
        except Exception as e:
            logger.error(f"Failed to convert message to row: {e}")
            raise ValidationError(f"Failed to convert message to row: {e}") from e

    def _row_to_thread(self, row: sqlite3.Row, messages: List[Message]) -> Thread:
        """Convert database row to Thread object.

        Args:
            row: Database row containing thread data
            messages: List of messages belonging to the thread

        Returns:
            Thread object

        Raises:
            ValueError: If row data is invalid
        """
        try:
            thread = Thread(
                id=row["id"],
                metadata=self._deserialize_metadata(row["metadata"]),
            )
            thread.created_at = datetime.fromisoformat(row["created_at"])
            thread.updated_at = datetime.fromisoformat(row["updated_at"])
            thread.messages = messages
            return thread
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to convert row to thread: {e}")
            raise ValueError(f"Failed to convert row to thread: {e}") from e

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """Convert database row to Message object.

        Args:
            row: Database row containing message data

        Returns:
            Message object

        Raises:
            ValueError: If row data is invalid
        """
        try:
            msg = Message(
                id=row["id"],
                content=row["content"],
                role=row["role"],
                metadata=self._deserialize_metadata(row["metadata"]),
                token_count=row["token_count"],
            )
            msg.created_at = datetime.fromisoformat(row["created_at"])
            return msg
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to convert row to message: {e}")
            raise ValueError(f"Failed to convert row to message: {e}") from e

    def save_thread(self, thread: Thread) -> None:
        """
        Save or update a thread.

        Args:
            thread (Thread): The thread to save

        Raises:
            ValidationError: If thread is invalid
            DatabaseConnectionError: If database connection fails
            DatabaseOperationError: If database operation fails
            DatabaseIntegrityError: If database integrity constraint is violated
        """
        if not thread:
            raise ValidationError("Thread cannot be None")

        if not thread.id:
            raise ValidationError("Thread ID cannot be empty")

        conn = None
        try:
            conn = self._get_connection()
            conn.execute("BEGIN TRANSACTION")

            # Save thread
            thread_row = self._thread_to_row(thread)
            conn.execute(SQLiteSchema.INSERT_THREAD, thread_row)

            # Delete existing messages if any
            conn.execute(SQLiteSchema.DELETE_THREAD_MESSAGES, (thread.id,))

            # Save messages with their order preserved
            messages = thread.messages
            if self.max_messages is not None and len(messages) > self.max_messages:
                messages = messages[-self.max_messages :]

            # Save messages
            for i, msg in enumerate(messages):
                msg_row = self._message_to_row(msg, thread.id, i)
                conn.execute(SQLiteSchema.INSERT_MESSAGE, msg_row)

            conn.commit()
            logger.debug(f"Saved thread {thread.id} with {len(messages)} messages")

        except sqlite3.IntegrityError as e:
            if conn:
                conn.rollback()
            logger.error(
                f"Database integrity error while saving thread {thread.id}: {e}"
            )
            raise DatabaseIntegrityError(
                f"Database integrity error while saving thread: {e}"
            ) from e
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error while saving thread {thread.id}: {e}")
            raise DatabaseOperationError(
                f"Database error while saving thread: {e}"
            ) from e
        except ValueError as e:
            if conn:
                conn.rollback()
            logger.error(f"Value error while saving thread {thread.id}: {e}")
            raise ValidationError(f"Invalid data while saving thread: {e}") from e
        finally:
            if conn:
                conn.close()

    def get_thread(
        self, thread_id: str, message_limit: Optional[int] = None
    ) -> Optional[Thread]:
        """
        Retrieve a thread by ID.

        Args:
            thread_id (str): ID of the thread to retrieve
            message_limit (Optional[int]): Maximum number of most recent messages to return.
                If None, return all stored messages.

        Returns:
            Optional[Thread]: Thread if found, None otherwise

        Raises:
            ValidationError: If thread_id is invalid
            DatabaseOperationError: If database operation fails
        """
        if not thread_id:
            raise ValidationError("Thread ID cannot be empty")

        if message_limit is not None and message_limit <= 0:
            raise ValidationError("message_limit must be a positive integer")

        conn = None
        try:
            conn = self._get_connection()

            # Get thread
            thread_row = conn.execute(SQLiteSchema.GET_THREAD, (thread_id,)).fetchone()
            if not thread_row:
                logger.debug(f"Thread {thread_id} not found")
                return None

            # Get messages
            if message_limit is not None:
                messages_rows = conn.execute(
                    SQLiteSchema.GET_THREAD_MESSAGES, (thread_id, message_limit)
                ).fetchall()
            else:
                messages_rows = conn.execute(
                    SQLiteSchema.GET_ALL_THREAD_MESSAGES, (thread_id,)
                ).fetchall()

            # Convert rows to messages
            messages = [self._row_to_message(row) for row in messages_rows]
            messages.reverse()  # Reverse to get chronological order

            # Convert row to thread
            thread = self._row_to_thread(thread_row, messages)
            return thread

        except sqlite3.Error as e:
            logger.error(f"Database error while retrieving thread {thread_id}: {e}")
            raise DatabaseOperationError(
                f"Database error while retrieving thread: {e}"
            ) from e
        finally:
            if conn:
                conn.close()

    def list_threads(self, limit: int = 100, offset: int = 0) -> List[Thread]:
        """
        List threads with pagination.

        Args:
            limit (int): Maximum number of threads to return
            offset (int): Number of threads to skip

        Returns:
            List[Thread]: List of threads

        Raises:
            ValidationError: If pagination parameters are invalid
            DatabaseOperationError: If database operation fails
        """
        if limit <= 0:
            raise ValidationError("Limit must be a positive integer")

        if offset < 0:
            raise ValidationError("Offset must be a non-negative integer")

        conn = None
        try:
            conn = self._get_connection()
            rows = conn.execute(SQLiteSchema.LIST_THREADS, (limit, offset)).fetchall()
            threads = []
            for row in rows:
                thread = self._row_to_thread(row, [])
                threads.append(thread)
            return threads
        except sqlite3.Error as e:
            logger.error(f"Database error while listing threads: {e}")
            raise DatabaseOperationError(
                f"Database error while listing threads: {e}"
            ) from e
        finally:
            if conn:
                conn.close()

    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete a thread.

        Args:
            thread_id (str): ID of the thread to delete

        Returns:
            bool: True if deleted, False if thread not found

        Raises:
            ValidationError: If thread_id is invalid
            DatabaseOperationError: If database operation fails
        """
        if not thread_id:
            raise ValidationError("Thread ID cannot be empty")

        conn = None
        try:
            conn = self._get_connection()
            conn.execute("BEGIN TRANSACTION")

            # Check if thread exists
            thread_row = conn.execute(SQLiteSchema.GET_THREAD, (thread_id,)).fetchone()
            if not thread_row:
                logger.debug(f"Thread {thread_id} not found for deletion")
                return False

            # Delete messages first (due to foreign key constraint)
            conn.execute(SQLiteSchema.DELETE_THREAD_MESSAGES, (thread_id,))

            # Delete thread
            conn.execute(SQLiteSchema.DELETE_THREAD, (thread_id,))
            conn.commit()
            logger.debug(f"Deleted thread {thread_id}")
            return True

        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error while deleting thread {thread_id}: {e}")
            raise DatabaseOperationError(
                f"Database error while deleting thread: {e}"
            ) from e
        finally:
            if conn:
                conn.close()

    def search_threads(self, query: Dict[str, Any]) -> List[Thread]:
        """
        Search for threads matching criteria.

        Args:
            query (Dict[str, Any]): Search criteria

        Returns:
            List[Thread]: List of matching threads

        Raises:
            ValidationError: If query is invalid
            DatabaseOperationError: If database operation fails
        """
        if not query:
            raise ValidationError("Search query cannot be empty")

        conditions = []
        params = []
        conn = None

        try:
            if "metadata" in query:
                if not isinstance(query["metadata"], dict):
                    raise ValidationError(
                        "Metadata search criteria must be a dictionary"
                    )
                for key, value in query["metadata"].items():
                    conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                    params.append(str(value))

            # Add content search across messages
            if "content" in query:
                if not isinstance(query["content"], str):
                    raise ValidationError("Content search criteria must be a string")
                conditions.append(
                    """
                    id IN (
                        SELECT DISTINCT thread_id 
                        FROM messages 
                        WHERE content LIKE ?
                    )
                    """
                )
                params.append(f"%{query['content']}%")

            # Build the SQL query
            if not conditions:
                raise ValidationError("No valid search criteria provided")

            sql = f"""
                SELECT * FROM threads 
                WHERE {' AND '.join(conditions)}
                ORDER BY updated_at DESC
            """

            try:
                conn = self._get_connection()
                thread_rows = conn.execute(sql, params).fetchall()

                threads = []
                for thread_row in thread_rows:
                    try:
                        thread = self.get_thread(thread_row["id"])
                        if thread:
                            threads.append(thread)
                    except Exception as e:
                        # Log error but continue with other threads
                        logger.error(
                            f"Error retrieving thread {thread_row['id']} during search: {e}"
                        )
                        continue

                logger.debug(f"Search found {len(threads)} threads matching query")
                return threads

            finally:
                if conn:
                    conn.close()

        except sqlite3.Error as e:
            logger.error(f"Database error during thread search: {e}")
            raise DatabaseOperationError(
                f"Database error during thread search: {e}"
            ) from e
        except ValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            logger.error(f"Error during thread search: {e}")
            raise DatabaseOperationError(f"Error during thread search: {e}") from e
