"""SQLite storage backend for MemexLLM."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiosqlite

from ..core.models import Message, Thread
from .base import BaseStorage


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
        """
        super().__init__(max_messages=max_messages)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute(SQLiteSchema.CREATE_THREADS_TABLE)
            conn.execute(SQLiteSchema.CREATE_MESSAGES_TABLE)
            conn.execute(SQLiteSchema.CREATE_MESSAGE_INDEX)
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper configuration.

        Returns:
            A configured SQLite connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable foreign key support
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> str:
        """Serialize metadata to JSON string.

        Args:
            metadata: Dictionary of metadata

        Returns:
            JSON string representation
        """
        return json.dumps(metadata)

    def _deserialize_metadata(self, metadata_str: str) -> Dict[str, Any]:
        """Deserialize metadata from JSON string.

        Args:
            metadata_str: JSON string of metadata

        Returns:
            Dictionary of metadata
        """
        result: Dict[str, Any] = json.loads(metadata_str)
        return result

    def _thread_to_row(self, thread: Thread) -> Tuple:
        """Convert Thread object to database row values.

        Args:
            thread: Thread to convert

        Returns:
            Tuple of values for database insertion
        """
        return (
            thread.id,
            thread.created_at.isoformat(),
            thread.updated_at.isoformat(),
            self._serialize_metadata(thread.metadata),
        )

    def _message_to_row(self, msg: Message, thread_id: str, index: int) -> Tuple:
        """Convert Message object to database row values.

        Args:
            msg: Message to convert
            thread_id: ID of the parent thread
            index: Position of message in thread

        Returns:
            Tuple of values for database insertion
        """
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

    def _row_to_thread(self, row: sqlite3.Row, messages: List[Message]) -> Thread:
        """Convert database row to Thread object.

        Args:
            row: Database row containing thread data
            messages: List of messages belonging to the thread

        Returns:
            Thread object
        """
        thread = Thread(
            id=row["id"],
            metadata=self._deserialize_metadata(row["metadata"]),
        )
        thread.created_at = datetime.fromisoformat(row["created_at"])
        thread.updated_at = datetime.fromisoformat(row["updated_at"])
        thread.messages = messages
        return thread

    def _row_to_message(self, row: sqlite3.Row) -> Message:
        """Convert database row to Message object.

        Args:
            row: Database row containing message data

        Returns:
            Message object
        """
        msg = Message(
            id=row["id"],
            content=row["content"],
            role=row["role"],
            metadata=self._deserialize_metadata(row["metadata"]),
            token_count=row["token_count"],
        )
        msg.created_at = datetime.fromisoformat(row["created_at"])
        return msg

    def save_thread(self, thread: Thread) -> None:
        """Save or update a thread and its messages.

        If max_messages is set, only stores the most recent messages up to max_messages.

        Args:
            thread: Thread to save
        """
        with self._get_connection() as conn:
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
        """
        with self._get_connection() as conn:
            # Get thread data
            thread_row = conn.execute(SQLiteSchema.GET_THREAD, (thread_id,)).fetchone()

            if not thread_row:
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
            return self._row_to_thread(thread_row, messages)

    def list_threads(self, limit: int = 100, offset: int = 0) -> List[Thread]:
        """List threads with pagination.

        Args:
            limit: Maximum number of threads to return
            offset: Number of threads to skip

        Returns:
            List of threads
        """
        with self._get_connection() as conn:
            thread_rows = conn.execute(
                SQLiteSchema.LIST_THREADS, (limit, offset)
            ).fetchall()

            threads = []
            for thread_row in thread_rows:
                thread = self.get_thread(thread_row["id"])
                if thread:
                    threads.append(thread)

            return threads

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread and its messages.

        Messages are automatically deleted due to CASCADE constraint.

        Args:
            thread_id: ID of the thread to delete

        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.execute(SQLiteSchema.DELETE_THREAD, (thread_id,))
            conn.commit()
            return cursor.rowcount > 0

    def search_threads(self, query: Dict[str, Any]) -> List[Thread]:
        """Search for threads matching criteria"""
        conditions = []
        params = []

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

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            thread_rows = conn.execute(sql, params).fetchall()

            return [
                thread
                for thread in (self.get_thread(row["id"]) for row in thread_rows)
                if thread is not None
            ]
