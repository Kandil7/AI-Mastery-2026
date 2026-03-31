#!/usr/bin/env python3
"""
Script to commit all modified files individually with descriptive commit messages.
"""

import subprocess
import sys
import os
from pathlib import Path


def get_modified_files():
    """Get list of modified files from git status."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        modified_files = []
        for line in lines:
            if line.strip():
                # Parse the git status output (first two characters are status codes)
                status = line[:2].strip()
                filename = line[3:].strip()
                # Only include modified files (M), added files (A), deleted files (D), etc.
                if status:
                    modified_files.append((status, filename))
        return modified_files
    except subprocess.CalledProcessError as e:
        print(f"Error getting git status: {e}")
        return []


def get_file_extension(filepath):
    """Get the file extension for a given filepath."""
    return Path(filepath).suffix.lower()


def generate_commit_message(status, filepath):
    """Generate a descriptive commit message based on file type and status."""
    ext = get_file_extension(filepath)
    
    # Determine the type of change
    if 'M' in status:
        change_type = "Update"
    elif 'A' in status:
        change_type = "Add"
    elif 'D' in status:
        change_type = "Delete"
    else:
        change_type = "Modify"
    
    # Determine file category
    code_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.kts'}
    doc_extensions = {'.md', '.txt', '.rst', '.adoc', '.tex'}
    config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.xml', '.html', '.css', '.scss', '.env'}
    
    if ext in code_extensions:
        file_category = "code"
    elif ext in doc_extensions:
        file_category = "documentation"
    elif ext in config_extensions:
        file_category = "configuration"
    else:
        file_category = "miscellaneous"
    
    return f"{change_type} {file_category}: {os.path.basename(filepath)}"


def commit_file(status, filepath):
    """Commit a single file with a generated commit message."""
    try:
        # Stage the file
        subprocess.run(["git", "add", filepath], check=True)
        
        # Generate commit message
        commit_msg = generate_commit_message(status, filepath)
        
        # Commit the file
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        
        print(f"Committed: {filepath} - {commit_msg}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error committing {filepath}: {e}")
        return False


def main():
    """Main function to commit all modified files individually."""
    # Change to the script's directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("Checking for modified files...")
    modified_files = get_modified_files()
    
    if not modified_files:
        print("No modified files to commit.")
        return
    
    print(f"Found {len(modified_files)} modified files:")
    for _, filepath in modified_files:
        print(f"  - {filepath}")
    
    print("\nCommitting each file individually...")
    success_count = 0
    
    for status, filepath in modified_files:
        if commit_file(status, filepath):
            success_count += 1
    
    print(f"\nSuccessfully committed {success_count} out of {len(modified_files)} files.")


if __name__ == "__main__":
    main()