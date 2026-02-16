"""
Database Backup Script
======================
Automated PostgreSQL backup to local/S3.

نسخ احتياطي لقاعدة البيانات
"""

import os
import sys
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

from src.core.config import settings


# ============================================================================
# Configuration
# ============================================================================

BACKUP_DIR = Path(os.environ.get("BACKUP_DIR", "./backups"))
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

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


def backup_to_local() -> tuple[bool, str]:
    """
    Backup PostgreSQL database to local directory.
    
    Uses pg_dump with compression.
    
    Returns:
        Tuple of (success, backup_path or error)
    """
    _log("Starting PostgreSQL backup to local...")
    
    # Create backup filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{DB_NAME}_backup_{timestamp}.sql.gz"
    backup_path = BACKUP_DIR / filename
    
    # Set PGPASSWORD for pg_dump
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASSWORD
    
    try:
        # Run pg_dump
        cmd = [
            "pg_dump",
            f"-h{DB_HOST}",
            f"-p{DB_PORT}",
            f"-U{DB_USER}",
            f"-d{DB_NAME}",
            "-Fc",  # Custom format with compression
            "--compress=9",
            f"--file={backup_path}"]
        
        success, output = _run_command(cmd)
        
        if not success:
            return False, output
        
        # Verify backup file exists and has size
        if not backup_path.exists():
            return False, f"Backup file not created: {backup_path}"
        
        file_size = backup_path.stat().st_size
        if file_size == 0:
            return False, f"Backup file is empty: {backup_path}"`
        
        _log(f"✅ Backup completed successfully")
        _log(f"   Path: {backup_path}")
        _log(f"   Size: {file_size:,} bytes")
        
        return True, str(backup_path)
    
    except Exception as e:
        return False, f"Backup failed: {str(e)}"


def backup_to_s3() -> tuple[bool, str]:
    """
    Backup PostgreSQL database to AWS S3.
    
    Requires:
    - AWS CLI installed and configured
    - S3 bucket accessible
    - Environment variables set
    
    Returns:
        Tuple of (success, S3 URI or error)
    """
    _log("Starting PostgreSQL backup to AWS S3...")
    
    if not S3_BUCKET:
        return False, "S3_BUCKET environment variable not set"
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{DB_NAME}_backup_{timestamp}.sql.gz"
    s3_uri = f"s3://{S3_BUCKET}/{filename}"
    
    # First backup to local
    local_success, local_path = backup_to_local()
    
    if not local_success:
        return False, "Local backup failed, cannot upload to S3"
    
    try:
        # Upload to S3 using AWS CLI
        cmd = [
            "aws", "s3", "cp",
            str(local_path),
            s3_uri,
        ]
        
        success, output = _run_command(cmd)
        
        if not success:
            return False, f"S3 upload failed: {output}"
        
        _log(f"✅ Backup uploaded to S3")
        _log(f"   S3 URI: {s3_uri}")
        _log(f"   Local file: {local_path}")
        
        return True, s3_uri
    
    except Exception as e:
        return False, f"S3 backup failed: {str(e)}"


def verify_backup(backup_path: Path) -> tuple[bool, str]:
    """
    Verify backup integrity by checking file and attempting restore.
    
    Args:
        backup_path: Path to backup file
        
    Returns:
        Tuple of (success, message)
    """
    _log(f"Verifying backup integrity: {backup_path}...")
    
    if not backup_path.exists():
        return False, f"Backup file not found: {backup_path}"`
    
    try:
        # Check file size
        file_size = backup_path.stat().st_size
        if file_size == 0:
            return False, f"Backup file is empty: {backup_path}"`
        
        # Attempt to restore (dry-run mode)
        cmd = [
            "psql",
            f"-h{DB_HOST}",
            f"-p{DB_PORT}",
            f"-U{DB_USER}",
            "-d", DB_NAME,
            "--single-transaction",
            "-c", f"\\i\\r\\n SET search_path TO temp;",  # Create temp database
            "-c", f"\\i\\r\\n \\i {backup_path};",  # Restore backup
            "-c", "\\i\\r\\n SELECT 1; DROP DATABASE IF EXISTS temp;",  # Check restore succeeded
        ]
        
        env = os.environ.copy()
        env["PGPASSWORD"] = DB_PASSWORD
        
        success, output = _run_command(cmd)
        
        if "SELECT 1" in output:
            _log(f"✅ Backup integrity verified successfully")
            return True, "Backup is valid and can be restored"
        else:
            _log(f"❌ Backup integrity check failed")
            return False, "Backup may be corrupted"
    
    except Exception as e:
        return False, f"Backup verification failed: {str(e)}"


def cleanup_old_backups(days_to_keep: int = 7) -> int:
    """
    Delete backup files older than N days.
    
    Args:
        days_to_keep: Number of days to keep (default: 7)
        
    Returns:
        Number of deleted files
    """
    _log(f"Cleaning up old backups (keeping last {days_to_keep} days)...")
    
    deleted_count = 0
    cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)
    
    try:
        for backup_file in BACKUP_DIR.glob(f"{DB_NAME}_backup_*.sql.gz"):
            # Get file modification time
            file_mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
            
            if file_mtime < cutoff_time:
                backup_file.unlink()
                deleted_count += 1
                _log(f"  Deleted: {backup_file.name}")
        
        _log(f"✅ Cleanup completed: {deleted_count} files deleted")
        return deleted_count
    
    except Exception as e:
        _log(f"❌ Cleanup failed: {str(e)}")
        return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Backup Script")
    parser.add_argument("--local", action="store_true", help="Backup to local directory (default)")
    parser.add_argument("--s3", action="store_true", help="Backup to AWS S3")
    parser.add_argument("--verify", type=str, help="Verify a specific backup file")
    parser.add_argument("--cleanup", type=int, default=0, help="Cleanup old backups (keep N days)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("   PostgreSQL Database Backup")
    print("=" * 60)
    print()
    
    # Handle operations
    if args.verify:
        backup_path = Path(args.verify)
        success, message = verify_backup(backup_path)
        print(f"Verification Result: {message}")
        sys.exit(0 if success else 1)
    
    if args.s3:
        success, result = backup_to_s3()
        if args.local:
            # Always create local copy when doing S3
            backup_to_local()
        print(f"S3 Backup Result: {result}")
        sys.exit(0 if success else 1)
    
    if args.local or not any([args.s3, args.verify, args.cleanup]):
        success, result = backup_to_local()
        print(f"Local Backup Result: {result}")
        sys.exit(0 if success else 1)
    
    if args.cleanup > 0:
        cleanup_old_backups(args.cleanup)
        sys.exit(0)
