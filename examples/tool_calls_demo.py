"""
Example demonstrating tool calls with OpenAI integration and history management.

This example shows:
1. How to work with tool calls in messages
2. How the history manager automatically handles tool calls and responses
3. How to maintain conversation context across multiple turns
"""

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from memexllm.algorithms import FIFOAlgorithm
from memexllm.core import HistoryManager
from memexllm.integrations.openai import with_history
from memexllm.storage import MemoryStorage


def print_separator(title: str) -> None:
    print(f"\n{'='*20} {title} {'='*20}")


def demonstrate_tool_calls() -> None:
    """
    Demonstrate how to use tool calls with the history manager.

    This example shows:
    1. Creating a client with history management
    2. Handling tool calls and responses
    3. Maintaining conversation context across multiple turns
    """
    # Skip this demo if no OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nSkipping demonstration - no OpenAI API key found")
        print("Please set the OPENAI_API_KEY environment variable to run this example")
        return

    print_separator("Demonstrating Tool Calls with History Management")

    # Create storage and history manager
    storage = MemoryStorage()
    algorithm = FIFOAlgorithm(max_messages=10)
    history_manager = HistoryManager(storage=storage, algorithm=algorithm)

    # Create and wrap OpenAI client with history management
    client = OpenAI()
    wrapped_client = with_history(history_manager=history_manager)(client)

    # Create a thread for our conversation
    thread = history_manager.create_thread(
        metadata={"purpose": "demonstrating tool calls"}
    )
    print(f"\nCreated thread with ID: {thread.id}")

    # Define a weather tool for the model to use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # Create a system message
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant that can check the weather.",
    }

    # Create a user message
    user_message = {
        "role": "user",
        "content": "What's the weather like in New York right now?",
    }

    print("\nUser: What's the weather like in New York right now?")

    # Send the initial messages
    initial_messages = [system_message, user_message]
    response = wrapped_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=initial_messages,
        tools=tools,
        thread_id=thread.id,
    )

    assistant_msg = response.choices[0].message
    print(f"\nA: {assistant_msg.content}")

    # Check if the model made a tool call
    if hasattr(assistant_msg, "tool_calls") and assistant_msg.tool_calls:
        tool_call = assistant_msg.tool_calls[0]
        print(f"[Tool Call: {tool_call.function.name}({tool_call.function.arguments})]")

        # In a real application, you would call the actual function here
        # For this example, we'll just return a hardcoded result
        weather_result = {
            "temperature": 72,
            "conditions": "sunny",
            "location": "New York",
        }

        # Send only the tool response - the history manager will automatically include
        # the assistant's message with the tool call in the correct order
        tool_response = {
            "role": "tool",
            "content": json.dumps(weather_result),
            "tool_call_id": tool_call.id,  # Use the actual tool call ID from the assistant's response
        }

        print("\n[Tool Response: Weather data for New York]")
        response = wrapped_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[tool_response], thread_id=thread.id
        )

        print(f"\nA: {response.choices[0].message.content}")

        # Now send a follow-up question that references the previous information
        follow_up_message = {
            "role": "user",
            "content": "Is that warmer than usual for this time of year?",
        }

        print("\nUser: Is that warmer than usual for this time of year?")
        response = wrapped_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[follow_up_message], thread_id=thread.id
        )

        print(f"\nA: {response.choices[0].message.content}")

        # Ask another follow-up question
        second_follow_up = {
            "role": "user",
            "content": "What should I wear for this weather?",
        }

        print("\nUser: What should I wear for this weather?")
        response = wrapped_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[second_follow_up], thread_id=thread.id
        )

        print(f"\nA: {response.choices[0].message.content}")

    # Show how the history manager has maintained the full conversation
    print_separator("Conversation History")
    thread = history_manager.get_thread(thread.id)
    print(f"\nThread contains {len(thread.messages)} messages")
    print(
        "\nThe history manager has automatically maintained the full conversation context,"
    )
    print("including the tool calls and responses.")
    print(
        "This allows the assistant to reference previous information in follow-up questions."
    )

    # Print all messages in the thread
    print_separator("Complete Message History")
    for i, msg in enumerate(thread.messages):
        print(f"\nMessage {i+1}:")
        print(f"Role: {msg.role}")
        print(f"Content: {msg.content}")

        # Print tool calls if present
        if msg.tool_calls:
            print("Tool Calls:")
            for tc in msg.tool_calls:
                print(f"  ID: {tc.id}")
                print(f"  Type: {tc.type}")
                print(f"  Function: {tc.function}")

        # Print tool call ID if present
        if msg.tool_call_id:
            print(f"Tool Call ID: {msg.tool_call_id}")

        # Print metadata
        print(f"Metadata: {msg.metadata}")


def main() -> None:
    """Run the demonstration."""
    demonstrate_tool_calls()


if __name__ == "__main__":
    main()
