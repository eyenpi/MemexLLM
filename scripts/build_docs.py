#!/usr/bin/env python3
"""Script to build documentation automatically."""
import os
import subprocess
import sys
from pathlib import Path


def build_docs() -> int:
    """Build the documentation using sphinx-build.

    Returns:
        int: 0 for success, 1 for failure
    """
    docs_dir = Path(__file__).parent.parent / "docs"
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"

    # Generate API documentation
    try:
        subprocess.run(
            [
                "sphinx-apidoc",
                "-o",
                str(source_dir),
                str(Path(__file__).parent.parent / "src" / "memexllm"),
                "--force",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error generating API documentation: {e}", file=sys.stderr)
        return 1

    # Build HTML documentation
    try:
        subprocess.run(
            [
                "sphinx-build",
                "-b",
                "html",
                str(source_dir),
                str(build_dir / "html"),
                # Don't treat warnings as errors for now
                # "-W",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    # Change to the project root directory
    os.chdir(Path(__file__).parent.parent)
    sys.exit(build_docs())
