"""
Example demonstrating OpenAI integration with SQLite storage and context management.

This example shows:
1. How messages are stored persistently in SQLite
2. How the algorithm controls context window for OpenAI
3. How storage and context limits work independently
"""

import os

from openai import OpenAI

from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager
from memexllm.integrations.openai import with_history
from memexllm.storage.sqlite import SQLiteStorage


def print_separator(title: str) -> None:
    print(f"\n{'='*20} {title} {'='*20}")


def main() -> None:
    # Initialize components
    storage = SQLiteStorage(
        db_path="chat_history.db",
        max_messages=1000,  # Store up to 1000 messages in SQLite
    )
    algorithm = FIFOAlgorithm(
        max_messages=4  # But only use last 4 messages for context
    )
    history_manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create and wrap OpenAI client
    client = OpenAI()  # Make sure OPENAI_API_KEY is set in environment
    wrapped_client = with_history(history_manager=history_manager)(client)

    # Create a new conversation thread
    thread = history_manager.create_thread(
        metadata={"topic": "counting", "user": "example"}
    )

    print_separator("Initial Conversation")

    # First exchange - establish context
    messages = [
        {
            "role": "system",
            "content": "You are a helpful counting assistant. Keep responses very brief.",
        },
        {"role": "user", "content": "Let's count from 1 to 5. Start with 1."},
    ]
    response = wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, thread_id=thread.id
    )
    print("\nUser: Let's count from 1 to 5. Start with 1.")
    print(f"Assistant: {response.choices[0].message.content}")

    # Continue counting, showing how context works
    for i in range(2, 6):
        messages = [{"role": "user", "content": f"Next number after {i-1}"}]
        response = wrapped_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, thread_id=thread.id
        )
        print(f"\nUser: Next number after {i-1}")
        print(f"Assistant: {response.choices[0].message.content}")

    # Show what's in storage vs what algorithm uses
    print_separator("Storage vs Algorithm")

    # Get full thread from storage directly
    raw_thread = storage.get_thread(thread.id)
    print("\nAll messages in SQLite storage:")
    for msg in raw_thread.messages:
        print(f"- [{msg.role}]: {msg.content}")

    # Get thread through manager (uses algorithm)
    context_thread = history_manager.get_thread(thread.id)
    print("\nMessages in algorithm context window:")
    for msg in context_thread.messages:
        print(f"- [{msg.role}]: {msg.content}")

    # Test if context window works by asking about earlier messages
    print_separator("Testing Context Window")

    messages = [{"role": "user", "content": "What was the first number we counted?"}]
    response = wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages, thread_id=thread.id
    )
    print("\nUser: What was the first number we counted?")
    print(f"Assistant: {response.choices[0].message.content}")
    print(
        "\n(Note: If the assistant doesn't remember '1', it's because it's outside the context window)"
    )

    # Clean up the database file
    print_separator("Cleanup")
    if os.path.exists("chat_history.db"):
        os.remove("chat_history.db")
        print("\nRemoved chat_history.db")


if __name__ == "__main__":
    main()
