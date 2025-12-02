# ============================================
# utils.py — Utility Functions
# ============================================
import hashlib
from pathlib import Path
from typing import Optional

def get_file_hash(file_path: Path) -> str:
    """Generate MD5 hash of file"""
    return hashlib.md5(file_path.read_bytes()).hexdigest()

def get_content_hash(text: str) -> str:
    """Generate SHA256 hash of text content"""
    return hashlib.sha256(text.encode()).hexdigest()

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix