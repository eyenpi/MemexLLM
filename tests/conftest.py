from typing import List

import pytest

from memexllm.core.models import Message, Thread


@pytest.fixture()  # type: ignore
def sample_thread() -> Thread:
    return Thread(
        id="test_thread",
        metadata={"user_id": "test_user"},
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ],
    )


@pytest.fixture()  # type: ignore
def sample_messages() -> List[Message]:
    return [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there"),
        Message(role="user", content="How are you?"),
        Message(role="assistant", content="I'm doing well!"),
    ]
