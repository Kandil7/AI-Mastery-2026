# File-by-File Git Commit Script

This script automates the process of committing each modified file individually with descriptive commit messages.

## Usage

1. Make sure you're in the repository directory
2. Run the script:
   ```
   python commit_files_individually.py
   ```

The script will:
- Detect all modified, added, or deleted files
- Generate appropriate commit messages based on file type and change
- Commit each file individually

## Commit Message Format

The script generates commit messages in the format:
- "Update code: filename" for modified code files
- "Add documentation: filename" for added documentation files
- "Delete configuration: filename" for deleted config files
- etc.

This follows conventional commit message practices while ensuring each file gets its own commit.