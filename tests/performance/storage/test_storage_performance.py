"""Performance tests for storage implementations."""

import os
import time
from typing import List, Optional

import pytest

from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager, Message, Thread
from memexllm.storage import MemoryStorage, SQLiteStorage
from memexllm.storage.sqlite import SQLiteSchema


def create_large_thread(num_messages: int) -> Thread:
    """Create a thread with specified number of messages."""
    thread = Thread(id="test")
    # Add some variety to message content to prevent SQLite query optimization
    for i in range(num_messages):
        content = f"Message {i} with some additional content to make it more realistic"
        thread.add_message(Message(content=content, role="user"))
    return thread


class TimingStats:
    """Helper class to track timing statistics."""

    def __init__(self) -> None:
        self.times: List[float] = []

    def add_time(self, time: float) -> None:
        self.times.append(time)

    @property
    def average(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0

    @property
    def min(self) -> float:
        return min(self.times) if self.times else 0

    @property
    def max(self) -> float:
        return max(self.times) if self.times else 0


def measure_retrieval_time(
    storage: SQLiteStorage,
    thread: Thread,
    algorithm: Optional[FIFOAlgorithm] = None,
    num_iterations: int = 5,
) -> TimingStats:
    """Measure time taken to retrieve thread with different configurations."""
    manager = HistoryManager(storage=storage, algorithm=algorithm)

    # First save the thread
    manager.storage.save_thread(thread)

    # Measure retrieval time
    stats = TimingStats()
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        manager.get_thread(thread.id)
        end_time = time.perf_counter()
        stats.add_time(end_time - start_time)

    return stats


@pytest.mark.parametrize("num_messages", [100, 1000, 10000])
def test_retrieval_performance_comparison(num_messages: int) -> None:
    """Compare performance of different retrieval strategies."""
    db_path = "perf_test.db"
    try:
        # Create test data
        thread = create_large_thread(num_messages)
        algorithm = FIFOAlgorithm(max_messages=10)  # Only want last 10 messages

        # Test unoptimized approach (get all then filter)
        unoptimized_storage = SQLiteStorage(db_path=db_path)
        unoptimized_stats = measure_retrieval_time(
            unoptimized_storage, thread, algorithm=None
        )

        # Test optimized approach (limit in query)
        optimized_storage = SQLiteStorage(db_path=db_path)
        optimized_stats = measure_retrieval_time(
            optimized_storage, thread, algorithm=algorithm
        )

        # Print results
        print(f"\nPerformance test with {num_messages} messages:")
        print("Unoptimized (get all messages):")
        print(f"  Average: {unoptimized_stats.average:.4f}s")
        print(f"  Min: {unoptimized_stats.min:.4f}s")
        print(f"  Max: {unoptimized_stats.max:.4f}s")

        print("\nOptimized (limit in query):")
        print(f"  Average: {optimized_stats.average:.4f}s")
        print(f"  Min: {optimized_stats.min:.4f}s")
        print(f"  Max: {optimized_stats.max:.4f}s")

        # Assert optimization is actually better
        assert optimized_stats.average < unoptimized_stats.average, (
            f"Optimized approach ({optimized_stats.average:.4f}s) should be faster "
            f"than unoptimized ({unoptimized_stats.average:.4f}s)"
        )

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


@pytest.mark.parametrize(
    "storage_class,storage_kwargs",
    [
        (SQLiteStorage, {"db_path": "perf_test.db"}),
        (MemoryStorage, {}),
    ],
)
def test_storage_scaling(storage_class: type, storage_kwargs: dict) -> None:
    """Test how different storage backends scale with message count."""
    try:
        message_counts = [100, 1000, 10000]
        algorithm = FIFOAlgorithm(max_messages=10)

        print(f"\nScaling test for {storage_class.__name__}:")
        for count in message_counts:
            storage = storage_class(**storage_kwargs)
            thread = create_large_thread(count)
            stats = measure_retrieval_time(storage, thread, algorithm=algorithm)

            print(f"\nMessage count: {count}")
            print(f"  Average retrieval time: {stats.average:.4f}s")
            print(f"  Min: {stats.min:.4f}s")
            print(f"  Max: {stats.max:.4f}s")

    finally:
        if "db_path" in storage_kwargs and os.path.exists(storage_kwargs["db_path"]):
            os.remove(storage_kwargs["db_path"])


def test_batch_operation_performance() -> None:
    """Test performance of batch operations."""
    db_path = "perf_test.db"
    try:
        storage = SQLiteStorage(db_path=db_path)
        batch_sizes = [1, 10, 100]
        num_messages = 1000

        print("\nBatch operation performance test:")
        for batch_size in batch_sizes:
            # Create threads in batches
            thread = create_large_thread(num_messages)

            start_time = time.perf_counter()
            with storage._get_connection() as conn:
                # Save thread
                conn.execute(SQLiteSchema.INSERT_THREAD, storage._thread_to_row(thread))

                # Save messages in batches
                for i in range(0, len(thread.messages), batch_size):
                    batch = thread.messages[i : i + batch_size]
                    conn.executemany(
                        SQLiteSchema.INSERT_MESSAGE,
                        [
                            storage._message_to_row(msg, thread.id, idx + i)
                            for idx, msg in enumerate(batch)
                        ],
                    )
                conn.commit()
            end_time = time.perf_counter()

            print(f"\nBatch size: {batch_size}")
            print(
                f"  Time to save {num_messages} messages: {end_time - start_time:.4f}s"
            )

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_search_performance() -> None:
    """Test search performance with different indexes."""
    db_path = "perf_test.db"
    try:
        storage = SQLiteStorage(db_path=db_path)
        num_threads = 100
        messages_per_thread = 100

        # Create test data
        print("\nCreating test data...")
        start_time = time.perf_counter()
        for i in range(num_threads):
            thread = Thread(
                id=f"thread_{i}",
                metadata={"category": "work" if i % 2 == 0 else "personal"},
            )
            for j in range(messages_per_thread):
                content = f"Message {j} in thread {i} about {'work' if i % 2 == 0 else 'personal'} stuff"
                thread.add_message(Message(content=content, role="user"))
            storage.save_thread(thread)
        setup_time = time.perf_counter() - start_time
        print(f"Setup time: {setup_time:.4f}s")

        # Test different search scenarios
        print("\nSearch performance test:")

        # Metadata search
        start_time = time.perf_counter()
        work_threads = storage.search_threads({"metadata": {"category": "work"}})
        metadata_time = time.perf_counter() - start_time
        print(f"Metadata search time: {metadata_time:.4f}s")
        print(f"Found {len(work_threads)} threads")

        # Content search
        start_time = time.perf_counter()
        work_content = storage.search_threads({"content": "work"})
        content_time = time.perf_counter() - start_time
        print(f"Content search time: {content_time:.4f}s")
        print(f"Found {len(work_content)} threads")

        # Combined search
        start_time = time.perf_counter()
        combined = storage.search_threads(
            {"metadata": {"category": "work"}, "content": "work"}
        )
        combined_time = time.perf_counter() - start_time
        print(f"Combined search time: {combined_time:.4f}s")
        print(f"Found {len(combined)} threads")

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
