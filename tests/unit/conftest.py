from typing import List, cast

import pytest

from memexllm.core import Message, MessageRole, Thread


@pytest.fixture()  # type: ignore
def sample_thread() -> Thread:
    return Thread(
        id="test_thread",
        metadata={"user_id": "test_user"},
        messages=[
            Message(role=cast(MessageRole, "user"), content="Hello"),
            Message(role=cast(MessageRole, "assistant"), content="Hi there"),
        ],
    )


@pytest.fixture()  # type: ignore
def sample_messages() -> List[Message]:
    return [
        Message(role=cast(MessageRole, "user"), content="Hello"),
        Message(role=cast(MessageRole, "assistant"), content="Hi there"),
        Message(role=cast(MessageRole, "user"), content="How are you?"),
        Message(role=cast(MessageRole, "assistant"), content="I'm doing well!"),
    ]
