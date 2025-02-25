# Upgrading MemexLLM

## For Maintainers: Release Process

### Creating a New Release

First bump version using poetry:

```bash
poetry version patch
```

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