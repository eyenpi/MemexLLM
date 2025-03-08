# Changelog

All notable changes to MemexLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-03-08

### Added
- Comprehensive exception handling and logging throughout the library
- Enhanced SQLite storage with improved error handling and logging
- OSS LLMOps stack example with LiteLLM, Langfuse, and MemexLLM integration

### Changed
- Standardized return type documentation across all methods
- Improved error handling in HistoryManager and storage modules
- Enhanced type hints and docstrings across storage and algorithm modules
- Fixed test_delete_thread to properly expect ThreadNotFoundError
- Improved test coverage for HistoryManager and SQLite storage

### Fixed
- Consistent error handling in get_thread method
- Proper error logging when a thread is not found
- Type annotations and docstrings now correctly reflect method behavior

## [0.0.7] - 2024-02-24

### Added
- Initial public release 