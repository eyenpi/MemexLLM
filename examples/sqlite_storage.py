"""
Example demonstrating how to use SQLite storage in MemexLLM.

This example shows:
1. Initializing SQLite storage with HistoryManager
2. Creating and managing conversation threads
3. Adding and retrieving messages
4. Listing and filtering conversations
5. Using metadata for organization

Installation:
    # Install the package with SQLite support using poetry
    poetry install
    poetry install --with sqlite

    # Or if using pip
    pip install "memexllm[sqlite]"

Running:
    # Using poetry
    poetry run python examples/sqlite_storage.py

    # Or directly with Python if installed in your environment
    python examples/sqlite_storage.py
"""

from pathlib import Path

from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager
from memexllm.storage.sqlite import SQLiteStorage


def main() -> None:
    # Initialize SQLite storage (will create a new database file if it doesn't exist)
    storage = SQLiteStorage(db_path="example.db")

    # Initialize history algorithm (optional)
    algorithm = FIFOAlgorithm(max_messages=50)

    # Create history manager
    history_manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create a new conversation thread about France
    france_thread = history_manager.create_thread(
        metadata={"topic": "geography", "language": "en"}
    )
    print(f"Created thread with ID: {france_thread.id}")

    # Add messages about France
    history_manager.add_message(
        thread_id=france_thread.id,
        content="What is the capital of France?",
        role="user",
        metadata={"language": "en"},
    )

    history_manager.add_message(
        thread_id=france_thread.id,
        content="The capital of France is Paris.",
        role="assistant",
        metadata={"confidence": 0.99},
    )

    # Create another thread about AI
    ai_thread = history_manager.create_thread(
        metadata={"topic": "AI", "language": "en"}
    )

    # Add messages about AI
    history_manager.add_message(
        thread_id=ai_thread.id,
        content="Tell me about machine learning.",
        role="user",
        metadata={"subtopic": "machine_learning"},
    )

    history_manager.add_message(
        thread_id=ai_thread.id,
        content="Machine learning is a subset of artificial intelligence...",
        role="assistant",
        metadata={"confidence": 0.95},
    )

    # Retrieve and display threads
    print("\nRetrieving France thread:")
    france_thread = history_manager.get_thread(france_thread.id)
    for msg in france_thread.messages:
        print(f"{msg.role}: {msg.content}")
        print(f"Metadata: {msg.metadata}")

    # List all threads
    print("\nListing all threads:")
    threads = history_manager.list_threads(limit=10)
    for thread in threads:
        print(f"Thread {thread.id}: {len(thread.messages)} messages")
        print(f"Thread metadata: {thread.metadata}")
    # Add follow-up messages
    history_manager.add_message(
        thread_id=france_thread.id,
        content="Is Paris also the largest city in France?",
        role="user",
    )

    history_manager.add_message(
        thread_id=france_thread.id,
        content="Yes, Paris is both the capital and largest city of France.",
        role="assistant",
        metadata={"confidence": 0.98},
    )

    # print all messages in the france thread
    print("\nPrinting all messages in the france thread:")
    for msg in france_thread.messages:
        print(f"{msg.role}: {msg.content}")
        print(f"Metadata: {msg.metadata}")

    # Delete the AI thread
    history_manager.delete_thread(ai_thread.id)
    print(f"\nDeleted thread: {ai_thread.id}")

    # Clean up the example database
    Path("example.db").unlink(missing_ok=True)
    print("\nCleaned up example database")


if __name__ == "__main__":
    main()
