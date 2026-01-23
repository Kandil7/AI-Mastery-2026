"""
Vector Database Backup and Recovery Module
==========================================
Automated backup, point-in-time recovery, and migration utilities.

Author: AI-Mastery-2026
"""

import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import json
import tarfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBBackupManager:
    """
    Manage backups and recovery for vector databases.
    
    Features:
    - Automated scheduled backups
    - Point-in-time recovery
    - Incremental backups
    - Backup retention policies
    """
    
    def __init__(self, db_path: str, backup_path: str = "./backups"):
        self.db_path = Path(db_path)
        self.backup_path = Path(backup_path)
        self.backup_path.mkdir(exist_ok=True)
        
        self.metadata_file = self.backup_path / "backup_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load backup metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"backups": [], "last_backup": None}
    
    def _save_metadata(self):
        """Save backup metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def create_backup(self, backup_type: str = "full", description: str = "") -> str:
        """
        Create a backup of the vector database.
        
        Args:
            backup_type: "full" or "incremental"
            description: Optional backup description
        
        Returns:
            Backup ID
        """
        timestamp = datetime.now()
        backup_id = f"backup_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        backup_dir = self.backup_path / backup_id
        
        logger.info(f"Creating {backup_type} backup: {backup_id}")
        
        try:
            if backup_type == "full":
                # Full backup: copy entire database
                shutil.copytree(self.db_path, backup_dir)
            elif backup_type == "incremental":
                # Incremental: only changed files since last backup
                self._create_incremental_backup(backup_dir)
            else:
                raise ValueError(f"Unknown backup type: {backup_type}")
            
            # Compress backup
            archive_path = self.backup_path / f"{backup_id}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(backup_dir, arcname=backup_id)
            
            # Remove uncompressed backup
            shutil.rmtree(backup_dir)
            
            # Calculate size
            size_mb = archive_path.stat().st_size / (1024 * 1024)
            
            # Update metadata
            backup_info = {
                "backup_id": backup_id,
                "timestamp": timestamp.isoformat(),
                "type": backup_type,
                "description": description,
                "size_mb": size_mb,
                "archive_path": str(archive_path)
            }
            
            self.metadata["backups"].append(backup_info)
            self.metadata["last_backup"] = backup_id
            self._save_metadata()
            
            logger.info(f"✅ Backup created: {backup_id} ({size_mb:.2f} MB)")
            return backup_id
        
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            raise
    
    def _create_incremental_backup(self, backup_dir: Path):
        """Create incremental backup (only changed files)."""
        if not self.metadata["last_backup"]:
            # No previous backup, do full
            shutil.copytree(self.db_path, backup_dir)
            return
        
        # Get timestamp of last backup
        last_backup_info = next(
            (b for b in self.metadata["backups"] 
             if b["backup_id"] == self.metadata["last_backup"]),
            None
        )
        
        if not last_backup_info:
            shutil.copytree(self.db_path, backup_dir)
            return
        
        last_backup_time = datetime.fromisoformat(last_backup_info["timestamp"])
        
        # Copy only files modified since last backup
        backup_dir.mkdir(exist_ok=True)
        for file_path in self.db_path.rglob("*"):
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime > last_backup_time:
                    rel_path = file_path.relative_to(self.db_path)
                    dest_path = backup_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_path)
    
    def restore_backup(self, backup_id: str, target_path: Optional[str] = None):
        """
        Restore from a backup.
        
        Args:
            backup_id: Backup to restore
            target_path: Optional target path (default: original db_path)
        """
        backup_info = next(
            (b for b in self.metadata["backups"] if b["backup_id"] == backup_id),
            None
        )
        
        if not backup_info:
            raise ValueError(f"Backup {backup_id} not found")
        
        archive_path = Path(backup_info["archive_path"])
        if not archive_path.exists():
            raise FileNotFoundError(f"Backup archive not found: {archive_path}")
        
        restore_path = Path(target_path) if target_path else self.db_path
        
        logger.info(f"Restoring backup {backup_id} to {restore_path}")
        
        # Extract archive
        temp_dir = self.backup_path / f"restore_{backup_id}"
        try:
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(self.backup_path)
            
            # Move to target location
            extracted_dir = self.backup_path / backup_id
            if restore_path.exists():
                shutil.rmtree(restore_path)
            shutil.move(str(extracted_dir), str(restore_path))
            
            logger.info(f"✅ Backup restored successfully")
        
        except Exception as e:
            logger.error(f"❌ Restore failed: {e}")
            raise
        
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def list_backups(self) -> List[Dict]:
        """List all available backups."""
        return self.metadata["backups"]
    
    def delete_backup(self, backup_id: str):
        """Delete a backup."""
        backup_info = next(
            (b for b in self.metadata["backups"] if b["backup_id"] == backup_id),
            None
        )
        
        if not backup_info:
            raise ValueError(f"Backup {backup_id} not found")
        
        # Delete archive
        archive_path = Path(backup_info["archive_path"])
        if archive_path.exists():
            archive_path.unlink()
        
        # Remove from metadata
        self.metadata["backups"] = [
            b for b in self.metadata["backups"] if b["backup_id"] != backup_id
        ]
        self._save_metadata()
        
        logger.info(f"Deleted backup: {backup_id}")
    
    def apply_retention_policy(self, days: int = 30):
        """
        Delete backups older than specified days.
        
        Args:
            days: Keep backups from last N days
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        backups_to_delete = [
            b for b in self.metadata["backups"]
            if datetime.fromisoformat(b["timestamp"]) < cutoff
        ]
        
        for backup_info in backups_to_delete:
            self.delete_backup(backup_info["backup_id"])
        
        logger.info(f"Retention policy applied: deleted {len(backups_to_delete)} old backups")
    
    def get_backup_stats(self) -> Dict:
        """Get backup statistics."""
        backups = self.metadata["backups"]
        
        if not backups:
            return {
                "total_backups": 0,
                "total_size_mb": 0,
                "oldest_backup": None,
                "newest_backup": None
            }
        
        total_size = sum(b["size_mb"] for b in backups)
        timestamps = [datetime.fromisoformat(b["timestamp"]) for b in backups]
        
        return {
            "total_backups": len(backups),
            "total_size_mb": total_size,
            "oldest_backup": min(timestamps).isoformat(),
            "newest_backup": max(timestamps).isoformat(),
            "last_backup": self.metadata["last_backup"]
        }


# ============================================================
# MIGRATION SCRIPTS
# ============================================================

def migrate_to_multi_tenant(old_db_path: str, new_db_path: str, 
                            default_tenant_id: str = "default"):
    """
    Migrate single-tenant vector DB to multi-tenant structure.
    
    Args:
        old_db_path: Path to old single-tenant database
        new_db_path: Path to new multi-tenant database
        default_tenant_id: Tenant ID for migrated data
    """
    from src.production.vector_db import MultiTenantVectorDB
    
    logger.info(f"Migrating {old_db_path} to multi-tenant structure")
    
    # Create new multi-tenant DB
    new_db = MultiTenantVectorDB(new_db_path)
    new_db.create_tenant(default_tenant_id)
    
    # Load old data
    old_db_path = Path(old_db_path)
    with open(old_db_path / 'vectors.pkl', 'rb') as f:
        old_data = pickle.load(f)
    
    # Migrate to new structure
    vectors = []
    metadata = []
    vector_ids = []
    
    for vid, (vec, meta) in old_data.items():
        vectors.append(vec)
        metadata.append(meta)
        vector_ids.append(vid)
    
    new_db.add_vectors(default_tenant_id, vectors, metadata, vector_ids)
    
    logger.info(f"✅ Migrated {len(vectors)} vectors to tenant '{default_tenant_id}'")


if __name__ == "__main__":
    # Example usage
    backup_manager = VectorDBBackupManager(
        db_path="./vector_db",
        backup_path="./backups"
    )
    
    # Create full backup
    backup_id = backup_manager.create_backup(
        backup_type="full",
        description="Daily automated backup"
    )
    print(f"Created backup: {backup_id}")
    
    # List backups
    print("\nAvailable backups:")
    for backup in backup_manager.list_backups():
        print(f"  {backup['backup_id']}: {backup['size_mb']:.2f} MB ({backup['timestamp']})")
    
    # Get stats
    stats = backup_manager.get_backup_stats()
    print(f"\nBackup stats:")
    print(f"  Total backups: {stats['total_backups']}")
    print(f"  Total size: {stats['total_size_mb']:.2f} MB")
    
    # Apply retention policy (keep last 30 days)
    backup_manager.apply_retention_policy(days=30)
    
    print("\n✅ Backup and recovery system ready!")
