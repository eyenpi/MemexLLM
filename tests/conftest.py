"""
Main conftest.py file that imports fixtures from other conftest files.
This allows tests to access fixtures from any conftest.py file.
"""

import os
import tempfile
from typing import Generator

# Add any global fixtures here
import pytest

# Import fixtures from unit tests
from tests.unit.conftest import sample_messages, sample_thread

# Import fixtures from integration tests if needed
# Uncomment if needed:
# from tests.integration.conftest import api_keys, integration_config

# Import fixtures from performance tests if needed
# Uncomment if needed:
# from tests.performance.conftest import performance_config, timer, benchmark


@pytest.fixture(scope="session")  # type: ignore
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir
