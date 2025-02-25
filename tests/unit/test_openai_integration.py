import asyncio
import builtins
import json
import unittest
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from memexllm.core.history import HistoryManager
from memexllm.core.models import (
    ImageContent,
    Message,
    MessageContent,
    TextContent,
    Thread,
    ToolCallContent,
)
from memexllm.integrations.openai import (
    _convert_content_to_openai_format,
    _convert_to_message,
    _convert_to_openai_messages,
    with_history,
)
from memexllm.storage.base import BaseStorage


class MockStorage(BaseStorage):
    """Mock storage for testing."""

    def __init__(self):
        self.threads = {}
        self.messages = {}

    def create_thread(self, thread_id=None, metadata=None):
        thread_id = thread_id or str(len(self.threads) + 1)
        self.threads[thread_id] = {
            "id": thread_id,
            "messages": [],
            "metadata": metadata or {},
        }
        return thread_id

    def get_thread(self, thread_id, message_limit=None):
        thread_data = self.threads.get(thread_id)
        if not thread_data:
            return None

        thread = Thread(id=thread_data["id"], metadata=thread_data.get("metadata", {}))
        messages = thread_data["messages"]

        # Apply message limit if specified
        if message_limit and len(messages) > message_limit:
            messages = messages[-message_limit:]

        for msg_data in messages:
            msg = Message(
                id=msg_data["id"],
                role=msg_data["role"],
                content=msg_data["content"],
                metadata=msg_data.get("metadata", {}),
            )
            thread.add_message(msg)
        return thread

    def save_thread(self, thread):
        """Save a thread to storage."""
        thread_dict = thread.to_dict()
        self.threads[thread.id] = {
            "id": thread.id,
            "messages": [],
            "metadata": thread_dict.get("metadata", {}),
        }

        # Add messages
        for msg_dict in thread_dict.get("messages", []):
            self.add_message(
                thread_id=thread.id,
                content=msg_dict["content"],
                role=msg_dict["role"],
                metadata=msg_dict.get("metadata", {}),
            )

        return thread.id

    def add_message(self, thread_id, content, role, metadata=None):
        if thread_id not in self.threads:
            self.create_thread(thread_id)

        message_id = str(len(self.messages) + 1)
        message = {
            "id": message_id,
            "thread_id": thread_id,
            "content": content,
            "role": role,
            "metadata": metadata or {},
        }
        self.messages[message_id] = message
        self.threads[thread_id]["messages"].append(message)
        return message_id

    def get_messages(self, thread_id, limit=None, offset=None):
        thread = self.threads.get(thread_id)
        if not thread:
            return []

        messages = thread["messages"]
        if offset:
            messages = messages[offset:]
        if limit:
            messages = messages[:limit]

        return [
            Message(
                id=msg["id"],
                role=msg["role"],
                content=msg["content"],
                metadata=msg.get("metadata", {}),
            )
            for msg in messages
        ]

    def update_thread(self, thread_id, metadata=None):
        if thread_id not in self.threads:
            return False

        if metadata:
            self.threads[thread_id]["metadata"] = metadata

        return True

    def delete_thread(self, thread_id):
        if thread_id not in self.threads:
            return False

        del self.threads[thread_id]
        # Delete associated messages
        self.messages = {
            k: v for k, v in self.messages.items() if v["thread_id"] != thread_id
        }
        return True

    def list_threads(self, limit=None, offset=None):
        threads = list(self.threads.values())

        if offset:
            threads = threads[offset:]
        if limit:
            threads = threads[:limit]

        return [Thread(id=t["id"], metadata=t.get("metadata", {})) for t in threads]

    def search_threads(self, query, limit=None, offset=None):
        """Search threads by query (mock implementation)."""
        # Simple mock implementation that returns all threads
        return self.list_threads(limit=limit, offset=offset)


class MockChatCompletion:
    """Mock ChatCompletion for testing."""

    def __init__(self, content="This is a response", model="gpt-4"):
        self.id = "chatcmpl-123456789"

        # Create a proper message structure that will work with _convert_to_message
        class MockChatCompletionMessage:
            def __init__(self, role, content):
                self.role = role
                self.content = content
                self.tool_calls = None
                self.function_call = None
                self.tool_call_id = None
                self.name = None

        # Create a choice with the message
        class MockChoice:
            def __init__(self, message, finish_reason="stop", index=0):
                self.message = message
                self.finish_reason = finish_reason
                self.index = index

        message = MockChatCompletionMessage("assistant", content)
        choice = MockChoice(message)

        self.choices = [choice]
        self.created = 1677858242
        self.model = model
        self.object = "chat.completion"
        self.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    def __await__(self):
        """Make the object awaitable for async tests."""

        async def _await_impl():
            return self

        return _await_impl().__await__()


# Create a patch for isinstance to make our mock pass the ChatCompletion check
original_isinstance = isinstance


def patched_isinstance(obj, classinfo):
    # Handle ChatCompletion check
    if (
        obj.__class__.__name__ == "MockChatCompletion"
        and getattr(classinfo, "__name__", "") == "ChatCompletion"
    ):
        return True

    # Handle ChatCompletionMessage check
    if (
        obj.__class__.__name__ == "MockChatCompletionMessage"
        and getattr(classinfo, "__name__", "") == "ChatCompletionMessage"
    ):
        return True

    return original_isinstance(obj, classinfo)


class TestOpenAIIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Apply the isinstance patch
        builtins.isinstance = patched_isinstance

    def tearDown(self):
        """Clean up after tests."""
        # Restore the original isinstance
        builtins.isinstance = original_isinstance

    def test_convert_simple_message(self):
        """Test converting a simple text message."""
        openai_msg = {"role": "user", "content": "Hello, world!"}
        msg = _convert_to_message(openai_msg)

        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello, world!")
        self.assertIsNone(msg.tool_calls)
        self.assertIsNone(msg.function_call)

    def test_convert_assistant_with_tool_calls(self):
        """Test converting an assistant message with tool calls."""
        openai_msg = {
            "role": "assistant",
            "content": "I'll search for that information.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps({"query": "weather in New York"}),
                    },
                }
            ],
        }

        msg = _convert_to_message(openai_msg)

        self.assertEqual(msg.role, "assistant")
        self.assertEqual(msg.content, "I'll search for that information.")
        self.assertIsNotNone(msg.tool_calls)
        self.assertEqual(len(msg.tool_calls), 1)
        self.assertEqual(msg.tool_calls[0].id, "call_123")
        self.assertEqual(msg.tool_calls[0].type, "function")
        self.assertEqual(msg.tool_calls[0].function["name"], "search")

    def test_convert_multimodal_message(self):
        """Test converting a message with multimodal content."""
        openai_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "high",
                    },
                },
            ],
        }

        msg = _convert_to_message(openai_msg)

        self.assertEqual(msg.role, "user")
        self.assertIsInstance(msg.content, list)
        self.assertEqual(len(msg.content), 2)
        self.assertIsInstance(msg.content[0], TextContent)
        self.assertEqual(msg.content[0].text, "What's in this image?")
        self.assertIsInstance(msg.content[1], ImageContent)
        self.assertEqual(msg.content[1].url, "https://example.com/image.jpg")
        self.assertEqual(msg.content[1].detail, "high")

    def test_convert_tool_message(self):
        """Test converting a tool message."""
        openai_msg = {
            "role": "tool",
            "content": '{"temperature": 72, "conditions": "sunny"}',
            "tool_call_id": "call_123",
        }

        msg = _convert_to_message(openai_msg)

        self.assertEqual(msg.role, "tool")
        self.assertEqual(msg.content, '{"temperature": 72, "conditions": "sunny"}')
        self.assertEqual(msg.tool_call_id, "call_123")

    def test_convert_to_openai_messages(self):
        """Test converting internal messages to OpenAI format."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!"),
            Message(
                role="assistant",
                content="How can I help you?",
                function_call={
                    "name": "get_weather",
                    "arguments": '{"location": "New York"}',
                },
            ),
            Message(
                role="function",
                content='{"temperature": 72, "conditions": "sunny"}',
                name="get_weather",
            ),
        ]

        openai_messages = _convert_to_openai_messages(messages)

        self.assertEqual(len(openai_messages), 4)
        self.assertEqual(openai_messages[0]["role"], "system")
        self.assertEqual(openai_messages[0]["content"], "You are a helpful assistant.")
        self.assertEqual(openai_messages[1]["role"], "user")
        self.assertEqual(openai_messages[1]["content"], "Hello!")
        self.assertEqual(openai_messages[2]["role"], "assistant")
        self.assertEqual(openai_messages[2]["content"], "How can I help you?")
        self.assertIn("function_call", openai_messages[2])
        self.assertEqual(openai_messages[3]["role"], "function")
        self.assertEqual(
            openai_messages[3]["content"], '{"temperature": 72, "conditions": "sunny"}'
        )
        self.assertEqual(openai_messages[3]["name"], "get_weather")

    def test_convert_multimodal_content(self):
        """Test converting multimodal content to OpenAI format."""
        content = [
            TextContent(text="Look at this image:"),
            ImageContent(url="https://example.com/image.jpg", detail="high"),
        ]

        openai_content = _convert_content_to_openai_format(content)

        self.assertIsInstance(openai_content, list)
        self.assertEqual(len(openai_content), 2)
        self.assertEqual(openai_content[0]["type"], "text")
        self.assertEqual(openai_content[0]["text"], "Look at this image:")
        self.assertEqual(openai_content[1]["type"], "image_url")
        self.assertEqual(
            openai_content[1]["image_url"]["url"], "https://example.com/image.jpg"
        )
        self.assertEqual(openai_content[1]["image_url"]["detail"], "high")

    @patch("openai.resources.chat.completions.Completions.create")
    def test_with_history_sync(self, mock_create):
        """Test the with_history decorator with a synchronous client."""
        # Setup mock response
        mock_create.return_value = MockChatCompletion()

        # Create storage and history manager
        storage = MockStorage()
        history_manager = HistoryManager(storage=storage)

        # Create client with history
        client = OpenAI()
        client = with_history(history_manager=history_manager)(client)

        # Test creating a new thread
        response = client.chat.completions.create(
            model="gpt-4", messages=[{"role": "user", "content": "Hello!"}]
        )

        # Verify the response
        self.assertEqual(response.choices[0].message.content, "This is a response")

        # Verify that messages were added to storage
        threads = list(storage.threads.values())
        self.assertEqual(len(threads), 1)
        thread = threads[0]
        self.assertEqual(
            len(thread["messages"]), 2
        )  # User message + assistant response
        self.assertEqual(thread["messages"][0]["role"], "user")
        self.assertEqual(thread["messages"][0]["content"], "Hello!")
        self.assertEqual(thread["messages"][1]["role"], "assistant")
        self.assertEqual(thread["messages"][1]["content"], "This is a response")

    @patch("openai.resources.chat.completions.Completions.create")
    def test_with_history_existing_thread(self, mock_create):
        """Test the with_history decorator with an existing thread."""
        # Setup mock response
        mock_create.return_value = MockChatCompletion()

        # Create storage and history manager
        storage = MockStorage()
        history_manager = HistoryManager(storage=storage)

        # Create a thread and add a system message
        thread_id = history_manager.create_thread().id
        history_manager.add_message(thread_id, "You are a helpful assistant.", "system")

        # Create client with history
        client = OpenAI()
        client = with_history(history_manager=history_manager)(client)

        # Test using existing thread
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "What's the weather?"}],
            thread_id=thread_id,
        )

        # Verify the response
        self.assertEqual(response.choices[0].message.content, "This is a response")

        # Verify that messages were added to storage
        thread = storage.threads[thread_id]
        self.assertEqual(len(thread["messages"]), 3)  # System + user + assistant
        self.assertEqual(thread["messages"][0]["role"], "system")
        self.assertEqual(thread["messages"][1]["role"], "user")
        self.assertEqual(thread["messages"][1]["content"], "What's the weather?")
        self.assertEqual(thread["messages"][2]["role"], "assistant")

    @patch("openai.resources.chat.completions.Completions.create")
    def test_with_history_system_message_override(self, mock_create):
        """Test system message override in with_history."""
        # Setup mock response
        mock_create.return_value = MockChatCompletion()

        # Create storage and history manager
        storage = MockStorage()
        history_manager = HistoryManager(storage=storage)

        # Create a thread and add a system message
        thread_id = history_manager.create_thread().id
        history_manager.add_message(thread_id, "You are a helpful assistant.", "system")

        # Create client with history
        client = OpenAI()
        client = with_history(history_manager=history_manager)(client)

        # Test with system message override
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a weather expert."},
                {"role": "user", "content": "What's the weather?"},
            ],
            thread_id=thread_id,
        )

        # Verify that the system message was overridden
        mock_create.assert_called_once()
        call_args = mock_create.call_args[1]
        messages = call_args["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a weather expert.")

    @patch("openai.resources.chat.completions.AsyncCompletions.create")
    def test_with_history_async(self, mock_create):
        """Test the with_history decorator with an async client."""
        # Setup mock response
        mock_create.return_value = MockChatCompletion()

        # Create storage and history manager
        storage = MockStorage()
        history_manager = HistoryManager(storage=storage)

        # Create client with history
        client = AsyncOpenAI()
        client = with_history(history_manager=history_manager)(client)

        # Run the async test in a new event loop
        async def run_test():
            # Test creating a new thread
            response = await client.chat.completions.create(
                model="gpt-4", messages=[{"role": "user", "content": "Hello!"}]
            )

            # Verify the response
            self.assertEqual(response.choices[0].message.content, "This is a response")

            # Verify that messages were added to storage
            threads = list(storage.threads.values())
            self.assertEqual(len(threads), 1)
            thread = threads[0]
            self.assertEqual(
                len(thread["messages"]), 2
            )  # User message + assistant response

        # Create and run a new event loop
        asyncio.run(run_test())

    def test_with_history_no_storage(self):
        """Test that with_history raises an error when no storage is provided."""
        client = OpenAI()
        with self.assertRaises(ValueError):
            with_history()(client)

    def test_with_history_existing_manager(self):
        """Test with_history with an existing HistoryManager."""
        storage = MockStorage()
        history_manager = HistoryManager(storage=storage)

        client = OpenAI()
        client = with_history(history_manager=history_manager)(client)

        # Just verify that it doesn't raise an error
        self.assertIsNotNone(client)


class TestModels(unittest.TestCase):
    """Test class specifically for the models module to improve coverage."""

    def test_message_from_dict_with_structured_content(self):
        """Test Message.from_dict with structured content."""
        message_data = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this image:"},
                {
                    "type": "image",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "high",
                    },
                },
            ],
            "id": "msg_123",
            "metadata": {"source": "test"},
            "token_count": 15,
        }

        message = Message.from_dict(message_data)

        self.assertEqual(message.id, "msg_123")
        self.assertEqual(message.role, "user")
        self.assertEqual(message.metadata, {"source": "test"})
        self.assertEqual(message.token_count, 15)
        self.assertIsInstance(message.content, list)
        self.assertEqual(len(message.content), 2)
        self.assertIsInstance(message.content[0], TextContent)
        self.assertEqual(message.content[0].text, "Look at this image:")
        self.assertIsInstance(message.content[1], ImageContent)
        self.assertEqual(message.content[1].url, "https://example.com/image.jpg")
        self.assertEqual(message.content[1].detail, "high")

    def test_message_from_dict_with_tool_calls(self):
        """Test Message.from_dict with tool calls."""
        message_data = {
            "role": "assistant",
            "content": "I'll search for that information.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps({"query": "weather in New York"}),
                    },
                }
            ],
        }

        message = Message.from_dict(message_data)

        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "I'll search for that information.")
        self.assertIsNotNone(message.tool_calls)
        self.assertEqual(len(message.tool_calls), 1)
        self.assertEqual(message.tool_calls[0].id, "call_123")
        self.assertEqual(message.tool_calls[0].type, "function")
        self.assertEqual(message.tool_calls[0].function["name"], "search")

    def test_message_from_dict_with_function_call(self):
        """Test Message.from_dict with function call."""
        message_data = {
            "role": "assistant",
            "content": "I'll get the weather for you.",
            "function_call": {
                "name": "get_weather",
                "arguments": json.dumps({"location": "New York"}),
            },
        }

        message = Message.from_dict(message_data)

        self.assertEqual(message.role, "assistant")
        self.assertEqual(message.content, "I'll get the weather for you.")
        self.assertIsNotNone(message.function_call)
        self.assertEqual(message.function_call["name"], "get_weather")
        self.assertEqual(
            message.function_call["arguments"], json.dumps({"location": "New York"})
        )

    def test_thread_serialize_content(self):
        """Test Thread._serialize_content method."""
        thread = Thread()

        # Test serializing string content
        string_content = "Hello, world!"
        serialized = thread._serialize_content(string_content)
        self.assertEqual(serialized, "Hello, world!")

        # Test serializing TextContent
        text_content = [TextContent(text="Hello, world!")]
        serialized = thread._serialize_content(text_content)
        self.assertEqual(serialized, [{"type": "text", "text": "Hello, world!"}])

        # Test serializing ImageContent with detail
        image_content = [
            ImageContent(url="https://example.com/image.jpg", detail="high")
        ]
        serialized = thread._serialize_content(image_content)
        self.assertEqual(
            serialized,
            [
                {
                    "type": "image",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "high",
                    },
                }
            ],
        )

        # Test serializing ImageContent without detail
        image_content = [ImageContent(url="https://example.com/image.jpg")]
        serialized = thread._serialize_content(image_content)
        self.assertEqual(
            serialized,
            [{"type": "image", "image_url": {"url": "https://example.com/image.jpg"}}],
        )

        # Test serializing ToolCallContent
        tool_call_content = [
            ToolCallContent(
                id="call_123",
                type="function",
                function={
                    "name": "search",
                    "arguments": json.dumps({"query": "weather"}),
                },
            )
        ]
        serialized = thread._serialize_content(tool_call_content)
        self.assertEqual(
            serialized,
            [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps({"query": "weather"}),
                    },
                }
            ],
        )

        # Test serializing mixed content
        mixed_content = [
            TextContent(text="Look at this:"),
            ImageContent(url="https://example.com/image.jpg", detail="high"),
        ]
        serialized = thread._serialize_content(mixed_content)
        self.assertEqual(
            serialized,
            [
                {"type": "text", "text": "Look at this:"},
                {
                    "type": "image",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "high",
                    },
                },
            ],
        )

    def test_thread_to_dict(self):
        """Test Thread.to_dict method."""
        # Create a thread with messages
        thread = Thread(id="thread_123", metadata={"source": "test"})

        # Add a simple text message
        text_message = Message(
            id="msg_1", role="user", content="Hello!", metadata={"source": "user_input"}
        )
        thread.add_message(text_message)

        # Add a message with tool calls
        tool_message = Message(
            id="msg_2",
            role="assistant",
            content="I'll search for that.",
            tool_calls=[
                ToolCallContent(
                    id="call_123",
                    type="function",
                    function={
                        "name": "search",
                        "arguments": json.dumps({"query": "weather"}),
                    },
                )
            ],
        )
        thread.add_message(tool_message)

        # Add a tool response message
        tool_response = Message(
            id="msg_3",
            role="tool",
            content=json.dumps({"temperature": 72, "conditions": "sunny"}),
            tool_call_id="call_123",
        )
        thread.add_message(tool_response)

        # Add a function message
        function_message = Message(
            id="msg_4",
            role="function",
            content=json.dumps({"result": "success"}),
            name="process_data",
        )
        thread.add_message(function_message)

        # Convert to dict
        thread_dict = thread.to_dict()

        # Verify the thread data
        self.assertEqual(thread_dict["id"], "thread_123")
        self.assertEqual(thread_dict["metadata"], {"source": "test"})
        self.assertIn("created_at", thread_dict)
        self.assertIn("updated_at", thread_dict)

        # Verify the messages
        self.assertEqual(len(thread_dict["messages"]), 4)

        # Check first message (text)
        self.assertEqual(thread_dict["messages"][0]["id"], "msg_1")
        self.assertEqual(thread_dict["messages"][0]["role"], "user")
        self.assertEqual(thread_dict["messages"][0]["content"], "Hello!")
        self.assertEqual(
            thread_dict["messages"][0]["metadata"], {"source": "user_input"}
        )

        # Check second message (with tool calls)
        self.assertEqual(thread_dict["messages"][1]["id"], "msg_2")
        self.assertEqual(thread_dict["messages"][1]["role"], "assistant")
        self.assertEqual(thread_dict["messages"][1]["content"], "I'll search for that.")
        self.assertEqual(len(thread_dict["messages"][1]["tool_calls"]), 1)

        # Check third message (tool response)
        self.assertEqual(thread_dict["messages"][2]["id"], "msg_3")
        self.assertEqual(thread_dict["messages"][2]["role"], "tool")
        self.assertEqual(
            thread_dict["messages"][2]["content"],
            json.dumps({"temperature": 72, "conditions": "sunny"}),
        )
        self.assertEqual(thread_dict["messages"][2]["tool_call_id"], "call_123")

        # Check fourth message (function)
        self.assertEqual(thread_dict["messages"][3]["id"], "msg_4")
        self.assertEqual(thread_dict["messages"][3]["role"], "function")
        self.assertEqual(
            thread_dict["messages"][3]["content"], json.dumps({"result": "success"})
        )
        self.assertEqual(thread_dict["messages"][3]["name"], "process_data")

    def test_thread_from_dict(self):
        """Test Thread.from_dict method."""
        # Create thread data
        now = datetime.now(timezone.utc)
        now_str = now.isoformat()

        thread_data = {
            "id": "thread_123",
            "metadata": {"source": "test"},
            "created_at": now_str,
            "updated_at": now_str,
            "messages": [
                {
                    "id": "msg_1",
                    "role": "user",
                    "content": "Hello!",
                    "created_at": now_str,
                    "metadata": {"source": "user_input"},
                },
                {
                    "id": "msg_2",
                    "role": "assistant",
                    "content": "I'll search for that.",
                    "created_at": now_str,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": json.dumps({"query": "weather"}),
                            },
                        }
                    ],
                },
                {
                    "id": "msg_3",
                    "role": "tool",
                    "content": json.dumps({"temperature": 72, "conditions": "sunny"}),
                    "created_at": now_str,
                    "tool_call_id": "call_123",
                },
            ],
        }

        # Create thread from dict
        thread = Thread.from_dict(thread_data)

        # Verify thread attributes
        self.assertEqual(thread.id, "thread_123")
        self.assertEqual(thread.metadata, {"source": "test"})
        self.assertEqual(thread.created_at.isoformat(), now_str)
        self.assertEqual(thread.updated_at.isoformat(), now_str)

        # Verify messages
        self.assertEqual(len(thread.messages), 3)

        # Check first message
        self.assertEqual(thread.messages[0].id, "msg_1")
        self.assertEqual(thread.messages[0].role, "user")
        self.assertEqual(thread.messages[0].content, "Hello!")
        self.assertEqual(thread.messages[0].metadata, {"source": "user_input"})
        self.assertEqual(thread.messages[0].created_at.isoformat(), now_str)

        # Check second message
        self.assertEqual(thread.messages[1].id, "msg_2")
        self.assertEqual(thread.messages[1].role, "assistant")
        self.assertEqual(thread.messages[1].content, "I'll search for that.")
        self.assertEqual(len(thread.messages[1].tool_calls), 1)
        self.assertEqual(thread.messages[1].tool_calls[0].id, "call_123")

        # Check third message
        self.assertEqual(thread.messages[2].id, "msg_3")
        self.assertEqual(thread.messages[2].role, "tool")
        self.assertEqual(
            thread.messages[2].content,
            json.dumps({"temperature": 72, "conditions": "sunny"}),
        )
        self.assertEqual(thread.messages[2].tool_call_id, "call_123")

    def test_thread_from_dict_minimal(self):
        """Test Thread.from_dict with minimal data."""
        thread_data = {"id": "thread_123"}

        thread = Thread.from_dict(thread_data)

        self.assertEqual(thread.id, "thread_123")
        self.assertEqual(thread.messages, [])
        self.assertEqual(thread.metadata, {})

    def test_thread_properties(self):
        """Test Thread properties and methods."""
        thread = Thread(id="thread_123")

        # Test initial state
        self.assertEqual(thread.message_count, 0)
        self.assertEqual(thread.get_messages(), [])

        # Add messages
        msg1 = Message(role="user", content="Hello")
        thread.add_message(msg1)

        # Test after adding a message
        self.assertEqual(thread.message_count, 1)
        self.assertEqual(thread.get_messages(), [msg1])

        # Add another message
        msg2 = Message(role="assistant", content="Hi there")
        thread.add_message(msg2)

        # Test after adding another message
        self.assertEqual(thread.message_count, 2)
        self.assertEqual(thread.get_messages(), [msg1, msg2])

        # Verify updated_at was changed
        self.assertGreater(thread.updated_at, thread.created_at)


if __name__ == "__main__":
    unittest.main()
