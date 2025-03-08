"""
Exceptions for the MemexLLM library.

This module defines custom exceptions used throughout the library to provide
clear error messages for common failure scenarios.
"""


class MemexLLMError(Exception):
    """Base exception for all MemexLLM errors."""

    pass


class ConfigurationError(MemexLLMError):
    """Exception raised for errors in the configuration."""

    pass


class APIError(MemexLLMError):
    """Exception raised for errors in API calls."""

    pass


class OpenAIAPIError(APIError):
    """Exception raised for errors in OpenAI API calls."""

    pass


class RateLimitError(APIError):
    """Exception raised when API rate limits are exceeded."""

    pass


class AuthenticationError(APIError):
    """Exception raised for authentication failures."""

    pass


class ResourceNotFoundError(MemexLLMError):
    """Exception raised when a requested resource is not found."""

    pass


class ThreadNotFoundError(ResourceNotFoundError):
    """Exception raised when a thread is not found."""

    pass


class MessageNotFoundError(ResourceNotFoundError):
    """Exception raised when a message is not found."""

    pass


class ValidationError(MemexLLMError):
    """Exception raised for validation errors."""

    pass


class StorageError(MemexLLMError):
    """Base exception for storage-related errors."""

    pass


class ConnectionError(StorageError):
    """Exception raised when connection to storage fails."""

    pass


class OperationError(StorageError):
    """Exception raised when a storage operation fails."""

    pass


class IntegrityError(StorageError):
    """Exception raised when a storage integrity constraint is violated."""

    pass
