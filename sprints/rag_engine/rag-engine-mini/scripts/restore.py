"""
Database Restore Script
=======================
Restore PostgreSQL database from backup.

استعادة قاعدة البيانات من النسخة الاحتياطي
"""

import os
import sys
import subprocess
from pathlib import Path

from src.core.config import settings


# ============================================================================
# Configuration
# ============================================================================

DB_NAME = "rag"
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")

S3_BUCKET = os.environ.get("S3_BUCKET", None)
S3_REGION = os.environ.get("S3_REGION", "us-east-1")


def _log(message: str) -> None:
    """Log message with timestamp."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def _run_command(command: list) -> tuple[bool, str]:
    """
    Run a shell command and return success status.

    Returns:
        Tuple of (success, output/error)
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, str(e)


def restore_from_local(backup_path: str) -> tuple[bool, str]:
    """
    Restore PostgreSQL database from local backup file.

    Args:
        backup_path: Path to .sql or .sql.gz backup file

    Returns:
        Tuple of (success, message)
    """
    _log(f"Starting PostgreSQL restore from local: {backup_path}...")

    # Verify backup file exists
    backup_file = Path(backup_path)
    if not backup_file.exists():
        return False, f"Backup file not found: {backup_path}"

    # Check if backup is compressed
    is_compressed = backup_file.suffix == ".gz"

    try:
        # Drop existing database (if any)
        _log("Dropping existing database...")
        drop_cmd = [
            "psql",
            f"-h{DB_HOST}",
            f"-p{DB_PORT}",
            f"-U{DB_USER}",
            "-d",
            DB_NAME,
            "-c",
            "DROP DATABASE IF EXISTS temp; DROP DATABASE IF EXISTS rag; CREATE DATABASE rag;",
        ]

        env = os.environ.copy()
        env["PGPASSWORD"] = DB_PASSWORD

        success, output = _run_command(drop_cmd)
        if not success:
            return False, f"Failed to drop database: {output}"

        _log("✅ Existing database dropped (or created)")

        # Restore backup
        if is_compressed:
            _log(f"Restoring compressed backup (gunzip)...")
            cat_cmd = [
                "gunzip",
                "-c",
                str(backup_file),
                "|",
                "psql",
                f"-h{DB_HOST}",
                f"-p{DB_PORT}",
                f"-U{DB_USER}",
                "-d",
                DB_NAME,
            ]
            success, output = _run_command(cat_cmd)
        else:
            _log(f"Restoring SQL backup...")
            restore_cmd = [
                "psql",
                f"-h{DB_HOST}",
                f"-p{DB_PORT}",
                f"-U{DB_USER}",
                "-d",
                DB_NAME,
                "-f",
                str(backup_file),
            ]
            success, output = _run_command(restore_cmd)

        if not success:
            return False, f"Restore failed: {output}"

        _log(f"✅ Database restored successfully from {backup_path}")
        return True, f"Database restored from {backup_path}"

    except Exception as e:
        return False, f"Restore failed: {str(e)}"


def restore_from_s3(backup_key: str, local_path: str = None) -> tuple[bool, str]:
    """
    Restore PostgreSQL database from S3 backup.

    Args:
        backup_key: S3 object key (e.g., rag_backup_20240115_123456.sql.gz)
        local_path: Optional local path to save before restoring (default: ./backups/)

    Returns:
        Tuple of (success, message)
    """
    _log(f"Starting PostgreSQL restore from S3: {backup_key}...")

    if not S3_BUCKET:
        return False, "S3_BUCKET environment variable not set"

    # Download from S3
    s3_uri = f"s3://{S3_BUCKET}/{backup_key}"
    _log(f"Downloading from S3: {s3_uri}...")

    try:
        download_cmd = ["aws", "s3", "cp", s3_uri]

        # Add local path if specified
        if local_path:
            download_cmd.append(local_path)
        else:
            local_path = Path("./backups") / backup_key
            local_path.parent.mkdir(parents=True, exist_ok=True)
            download_cmd.append(str(local_path))

        success, output = _run_command(download_cmd)
        if not success:
            return False, f"S3 download failed: {output}"

        _log(f"✅ Downloaded to: {local_path}")

        # Restore from local file
        return restore_from_local(str(local_path))

    except Exception as e:
        return False, f"S3 restore failed: {str(e)}"


def list_backups() -> list:
    """
    List all available backups (local and S3).

    Returns:
        List of backup metadata dicts
    """
    _log("Listing available backups...")

    backups = []

    # List local backups
    from pathlib import Path

    local_backups_dir = Path("./backups")

    if local_backups_dir.exists():
        for backup_file in local_backups_dir.glob("rag_backup_*.sql.gz"):
            stat = backup_file.stat()
            backups.append(
                {
                    "source": "local",
                    "path": str(backup_file),
                    "name": backup_file.name,
                    "size_bytes": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

    # List S3 backups (requires AWS CLI)
    try:
        s3_cmd = ["aws", "s3", "ls", f"s3://{S3_BUCKET}"]
        success, output = _run_command(s3_cmd)

        if success and output:
            for line in output.split("\n"):
                if backup_file.endswith(".sql.gz"):
                    backups.append(
                        {
                            "source": "s3",
                            "name": backup_file.split("/")[-1],
                            "path": f"s3://{S3_BUCKET}/{backup_file}",
                            "size_bytes": 0,  # AWS CLI doesn't show size easily
                            "created_at": "unknown",  # Would need head-object
                        }
                    )

        _log(f"Found {len(backups)} local backups")
        if S3_BUCKET:
            _log(f"Found {len([b for b in backups if b['source'] == 's3'])} S3 backups")
        return backups

    except Exception as e:
        _log(f"⚠️  Could not list S3 backups: {str(e)}")
        return backups


if __name__ == "__main__":
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Database Restore Script")
    parser.add_argument("--local", type=str, help="Restore from local backup file")
    parser.add_argument("--s3", type=str, help="Restore from S3 backup key")
    parser.add_argument("--list", action="store_true", help="List available backups")
    parser.add_argument("--download-s3", type=str, help="Download S3 backup to local")

    args = parser.parse_args()

    print("=" * 60)
    print("   PostgreSQL Database Restore")
    print("=" * 60)
    print()

    if args.list:
        backups = list_backups()

        if backups:
            print(f"Available Backups ({len(backups)}):")
            print(f"{'-' * 50}")
            for i, backup in enumerate(backups, 1):
                source_icon = "☁️" if backup["source"] == "s3" else "💾"
                size_mb = backup.get("size_bytes", 0) / (1024 * 1024)
                print(f"{i}. [{source_icon}] {backup['name'][:50]:...}")
                print(f"   Created: {backup.get('created_at', 'N/A')}")
                if size_mb > 0:
                    print(f"   Size: {size_mb:.2f} MB")
                print(f"   Path: {backup['path']}")
            print()
        else:
            print("No backups found")

        sys.exit(0)

    if args.download_s3:
        success, result = restore_from_s3(args.download_s3)
        print(f"Download Result: {result}")
        sys.exit(0 if success else 1)

    if args.local:
        success, result = restore_from_local(args.local)
        print(f"Restore Result: {result}")
        sys.exit(0 if success else 1)

    if args.s3:
        success, result = restore_from_s3(args.s3)
        print(f"S3 Restore Result: {result}")
        sys.exit(0 if success else 1)

    parser.print_help()
    sys.exit(1)
