# Upgrading MemexLLM

This guide provides instructions on how to upgrade your MemexLLM installation to newer versions.

## Using pip (Recommended)

To upgrade MemexLLM to the latest version, run:

```bash
pip install --upgrade memexllm
```

To install a specific version, run:

```bash
pip install memexllm==X.Y.Z  # Replace X.Y.Z with the desired version number
```

For example:
```bash
pip install memexllm==0.0.4
```

## Manual Installation from Source

If you installed MemexLLM from source, you can upgrade by following these steps:

1. Navigate to your MemexLLM directory:
   ```bash
   cd path/to/MemexLLM
   ```

2. Pull the latest changes:
   ```bash
   git pull origin main
   ```

3. Check out the specific version tag (optional):
   ```bash
   git checkout vX.Y.Z  # Replace X.Y.Z with the desired version number
   ```

4. Reinstall the package:
   ```bash
   pip install -e .
   ```

## Checking Current Version

You can check your currently installed version by running:

```bash
pip show memexllm
```

## Version History

For a complete list of changes in each version, please refer to our [GitHub Releases page](https://github.com/eyenpi/MemexLLM/releases).

## Important Notes

- Always backup your data before upgrading
- Check the release notes for any breaking changes
- Make sure to upgrade any dependencies if required
- If you encounter any issues, please [report them on GitHub](https://github.com/eyenpi/MemexLLM/issues)

## For Maintainers: Release Process

### Creating a New Release

1. Ensure all changes are committed and pushed to the `dev` branch
2. Update version numbers in relevant files (e.g., `setup.py`, `package.json`)
3. Create and push a new tag:
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```

### Managing Tags

If you need to update an existing tag:

1. Delete the tag locally and remotely:
   ```bash
   git tag -d vX.Y.Z
   git push origin :refs/tags/vX.Y.Z
   ```

2. Create the tag again:
   ```bash
   git tag -a vX.Y.Z -m "Release version X.Y.Z"
   git push origin vX.Y.Z
   ```

### Viewing Tags

- List all tags:
  ```bash
  git tag
  ```

- View tag details:
  ```bash
  git show vX.Y.Z
  ```

### Best Practices

- Use semantic versioning (MAJOR.MINOR.PATCH)
- Always include a descriptive message with the tag
- Create a GitHub release with detailed changelog after pushing the tag
- Test the release thoroughly before creating the tag 