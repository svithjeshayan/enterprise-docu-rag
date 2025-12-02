# ============================================
# database.py — Database Operations
# ============================================
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database operations"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.init_db()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def init_db(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            c = conn.cursor()
            
            # Users
            c.execute("""CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        name TEXT,
                        created_at TEXT,
                        last_active TEXT
            )""")
            
            # Chats
            c.execute("""CREATE TABLE IF NOT EXISTS chats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        timestamp TEXT,
                        role TEXT,
                        content TEXT,
                        FOREIGN KEY (user_id) REFERENCES users(user_id)
            )""")
            
            # Documents
            c.execute("""CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT UNIQUE,
                        file_hash TEXT,
                        content_hash TEXT,
                        processed_at TEXT,
                        chunk_count INTEGER,
                        status TEXT DEFAULT 'active',
                        error_message TEXT
            )""")
            
            # Rate limits
            c.execute("""CREATE TABLE IF NOT EXISTS rate_limits (
                        user_id TEXT,
                        timestamp TEXT,
                        PRIMARY KEY (user_id, timestamp)
            )""")
            
            # Indexes
            c.execute("CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, timestamp)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_docs_status ON documents(status)")
            
            conn.commit()
    
    def create_user(self, user_id: str, name: str) -> bool:
        """Create new user"""
        try:
            with self.get_connection() as conn:
                c = conn.cursor()
                c.execute("""INSERT INTO users (user_id, name, created_at, last_active) 
                            VALUES (?, ?, ?, ?)""",
                         (user_id, name, datetime.now().isoformat(), datetime.now().isoformat()))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False
    
    def update_user_activity(self, user_id: str):
        """Update user's last active timestamp"""
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute("UPDATE users SET last_active=? WHERE user_id=?",
                     (datetime.now().isoformat(), user_id))
            conn.commit()
    
    def save_message(self, user_id: str, role: str, content: str):
        """Save chat message"""
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute("""INSERT INTO chats (user_id, timestamp, role, content) 
                        VALUES (?, ?, ?, ?)""",
                     (user_id, datetime.now().isoformat(), role, content))
            conn.commit()
    
    def get_chat_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's chat history"""
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute("""SELECT role, content FROM chats 
                        WHERE user_id=? 
                        ORDER BY timestamp DESC 
                        LIMIT ?""", (user_id, limit))
            return [{"role": r[0], "content": r[1]} for r in reversed(c.fetchall())]
    
    def clear_chat_history(self, user_id: str):
        """Clear user's chat history"""
        with self.get_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM chats WHERE user_id=?", (user_id,))
            conn.commit()
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        with self.get_connection() as conn:
            c = conn.cursor()
            
            c.execute("SELECT COUNT(*) FROM users")
            total_users = c.fetchone()[0]
            
            c.execute("SELECT COUNT(*) FROM documents WHERE status='active'")
            active_docs = c.fetchone()[0]
            
            c.execute("SELECT SUM(chunk_count) FROM documents WHERE status='active'")
            total_chunks = c.fetchone()[0] or 0
            
            c.execute("SELECT COUNT(*) FROM chats")
            total_messages = c.fetchone()[0]
            
            return {
                "total_users": total_users,
                "active_documents": active_docs,
                "total_chunks": total_chunks,
                "total_messages": total_messages
            }

