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
    LIST_THREADS = """
        SELECT * FROM threads 
        ORDER BY updated_at DESC 
        LIMIT ? OFFSET ?
    """


class SQLiteStorage(BaseStorage):
    """SQLite storage backend for threads.

    This implementation uses SQLite for persistent storage of threads and messages.
    It maintains referential integrity between threads and messages using foreign keys.

    Attributes:
        db_path: Path to the SQLite database file
        max_messages: Maximum number of messages to store per thread
    """

    def __init__(
        self, db_path: str = "memexllm.db", max_messages: Optional[int] = None
    ):
        """Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file
            max_messages: Maximum number of messages to store per thread.
                If None, store all messages.

        Raises:
            DatabaseConnectionError: If the database cannot be initialized
        """
        super().__init__(max_messages=max_messages)

        # Validate input parameters
        if not db_path:
            raise ValueError("Database path cannot be empty")

        if max_messages is not None and max_messages <= 0:
            raise ValueError("max_messages must be a positive integer")

        self.db_path = db_path
        try:
            self._init_db()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseConnectionError(f"Failed to initialize database: {e}") from e

    def _init_db(self) -> None:
        """Initialize database schema.

        Raises:
            DatabaseConnectionError: If the database cannot be connected to
            DatabaseOperationError: If schema creation fails
        """
        try:
            # Ensure the directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

            with self._get_connection() as conn:
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
            ValueError: If metadata cannot be serialized
        """
        try:
            return json.dumps(metadata)
        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize metadata: {e}")
            raise ValueError(f"Failed to serialize metadata: {e}") from e

    def _deserialize_metadata(self, metadata_str: str) -> Dict[str, Any]:
        """Deserialize metadata from JSON string.

        Args:
            metadata_str: JSON string of metadata

        Returns:
            Dictionary of metadata

        Raises:
            ValueError: If metadata cannot be deserialized
        """
        try:
            result: Dict[str, Any] = json.loads(metadata_str)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to deserialize metadata: {e}")
            raise ValueError(f"Failed to deserialize metadata: {e}") from e

    def _thread_to_row(self, thread: Thread) -> Tuple:
        """Convert Thread object to database row values.

        Args:
            thread: Thread to convert

        Returns:
            Tuple of values for database insertion

        Raises:
            ValueError: If thread data is invalid
        """
        if not thread.id:
            raise ValueError("Thread ID cannot be empty")

        try:
            return (
                thread.id,
                thread.created_at.isoformat(),
                thread.updated_at.isoformat(),
                self._serialize_metadata(thread.metadata),
            )
        except Exception as e:
            logger.error(f"Failed to convert thread to row: {e}")
            raise ValueError(f"Failed to convert thread to row: {e}") from e

    def _message_to_row(self, msg: Message, thread_id: str, index: int) -> Tuple:
        """Convert Message object to database row values.

        Args:
            msg: Message to convert
            thread_id: ID of the parent thread
            index: Position of message in thread

        Returns:
            Tuple of values for database insertion

        Raises:
            ValueError: If message data is invalid
        """
        if not msg.id:
            raise ValueError("Message ID cannot be empty")

        if not thread_id:
            raise ValueError("Thread ID cannot be empty")

        if index < 0:
            raise ValueError("Message index must be non-negative")

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
            raise ValueError(f"Failed to convert message to row: {e}") from e

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
        """Save or update a thread and its messages.

        If max_messages is set, only stores the most recent messages up to max_messages.

        Args:
            thread: Thread to save

        Raises:
            ValueError: If thread is invalid
            DatabaseOperationError: If database operation fails
            DatabaseIntegrityError: If database integrity constraint is violated
        """
        if not thread:
            raise ValueError("Thread cannot be None")

        if not thread.id:
            raise ValueError("Thread ID cannot be empty")

        conn = None
        try:
            conn = self._get_connection()
            # Start a transaction
            conn.execute("BEGIN TRANSACTION")

            # Save thread
            conn.execute(SQLiteSchema.INSERT_THREAD, self._thread_to_row(thread))

            # Delete existing messages
            conn.execute(SQLiteSchema.DELETE_THREAD_MESSAGES, (thread.id,))

            # Save messages with their order preserved
            messages = thread.messages
            if self.max_messages is not None and len(messages) > self.max_messages:
                messages = messages[-self.max_messages :]

            for idx, msg in enumerate(messages):
                conn.execute(
                    SQLiteSchema.INSERT_MESSAGE,
                    self._message_to_row(msg, thread.id, idx),
                )
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
            raise
        finally:
            if conn:
                conn.close()

    def get_thread(
        self, thread_id: str, message_limit: Optional[int] = None
    ) -> Optional[Thread]:
        """Retrieve a thread by ID.

        Args:
            thread_id: ID of the thread to retrieve
            message_limit: Maximum number of most recent messages to return.
                If None, return all stored messages.

        Returns:
            Thread if found, None otherwise

        Raises:
            ValueError: If thread_id is invalid
            DatabaseOperationError: If database operation fails
        """
        if not thread_id:
            raise ValueError("Thread ID cannot be empty")

        if message_limit is not None and message_limit <= 0:
            raise ValueError("message_limit must be a positive integer")

        try:
            with self._get_connection() as conn:
                # Get thread data
                thread_row = conn.execute(
                    SQLiteSchema.GET_THREAD, (thread_id,)
                ).fetchone()

                if not thread_row:
                    logger.debug(f"Thread {thread_id} not found")
                    return None

                # Determine effective limit:
                # - If message_limit is set, use it
                # - Otherwise if max_messages is set, use it
                # - Otherwise no limit (-1)
                effective_limit = (
                    message_limit
                    if message_limit is not None
                    else self.max_messages if self.max_messages is not None else -1
                )

                # Get messages in order with limit
                query = """
                    SELECT * FROM messages 
                    WHERE thread_id = ? 
                    ORDER BY message_index DESC
                    LIMIT ?
                """
                msg_rows = conn.execute(query, (thread_id, effective_limit)).fetchall()

                # Convert rows to objects
                messages = [self._row_to_message(row) for row in msg_rows]
                messages.reverse()  # Reverse since we ordered DESC in query
                thread = self._row_to_thread(thread_row, messages)
                logger.debug(
                    f"Retrieved thread {thread_id} with {len(messages)} messages"
                )
                return thread

        except sqlite3.Error as e:
            logger.error(f"Database error while retrieving thread {thread_id}: {e}")
            raise DatabaseOperationError(
                f"Database error while retrieving thread: {e}"
            ) from e
        except ValueError as e:
            logger.error(f"Value error while retrieving thread {thread_id}: {e}")
            raise

    def list_threads(self, limit: int = 100, offset: int = 0) -> List[Thread]:
        """List threads with pagination.

        Args:
            limit: Maximum number of threads to return
            offset: Number of threads to skip

        Returns:
            List of threads

        Raises:
            ValueError: If pagination parameters are invalid
            DatabaseOperationError: If database operation fails
        """
        if limit <= 0:
            raise ValueError("Limit must be a positive integer")

        if offset < 0:
            raise ValueError("Offset must be a non-negative integer")

        try:
            with self._get_connection() as conn:
                thread_rows = conn.execute(
                    SQLiteSchema.LIST_THREADS, (limit, offset)
                ).fetchall()

                threads = []
                for thread_row in thread_rows:
                    try:
                        thread = self.get_thread(thread_row["id"])
                        if thread:
                            threads.append(thread)
                    except Exception as e:
                        # Log error but continue with other threads
                        logger.error(f"Error retrieving thread {thread_row['id']}: {e}")
                        continue

                logger.debug(
                    f"Listed {len(threads)} threads (limit={limit}, offset={offset})"
                )
                return threads

        except sqlite3.Error as e:
            logger.error(f"Database error while listing threads: {e}")
            raise DatabaseOperationError(
                f"Database error while listing threads: {e}"
            ) from e

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and its messages.

        Messages are automatically deleted due to CASCADE constraint.

        Args:
            thread_id: ID of the thread to delete

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: If thread_id is invalid
            DatabaseOperationError: If database operation fails
        """
        if not thread_id:
            raise ValueError("Thread ID cannot be empty")

        try:
            with self._get_connection() as conn:
                cursor = conn.execute(SQLiteSchema.DELETE_THREAD, (thread_id,))
                conn.commit()
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.debug(f"Deleted thread {thread_id}")
                else:
                    logger.debug(f"Thread {thread_id} not found for deletion")
                return deleted

        except sqlite3.Error as e:
            logger.error(f"Database error while deleting thread {thread_id}: {e}")
            raise DatabaseOperationError(
                f"Database error while deleting thread: {e}"
            ) from e

    def search_threads(self, query: Dict[str, Any]) -> List[Thread]:
        """Search for threads matching criteria

        Args:
            query: Search criteria

        Returns:
            List of matching threads

        Raises:
            ValueError: If query is invalid
            DatabaseOperationError: If database operation fails
        """
        if not query:
            raise ValueError("Search query cannot be empty")

        conditions = []
        params = []

        try:
            if "metadata" in query:
                for key, value in query["metadata"].items():
                    conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                    params.append(str(value))

            # Add content search across messages
            if "content" in query:
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

            sql = "SELECT * FROM threads"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)

            conn = None
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
