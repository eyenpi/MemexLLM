import os
from typing import Any, Dict, Generator

import pytest


@pytest.fixture()  # type: ignore
def api_keys() -> Dict[str, str]:
    """Fixture to provide API keys for integration tests."""
    return {
        "openai": os.environ.get("OPENAI_API_KEY", "dummy_key"),
    }


@pytest.fixture()  # type: ignore
def integration_config() -> Dict[str, Any]:
    """Fixture to provide configuration for integration tests."""
    return {
        "timeout": 30,
        "retry_attempts": 3,
        "base_url": os.environ.get("API_BASE_URL", "https://api.openai.com/v1"),
    }


@pytest.fixture(autouse=True)  # type: ignore
def setup_teardown() -> Generator[None, None, None]:
    """Setup and teardown for integration tests."""
    # Setup code
    yield
    # Teardown code
