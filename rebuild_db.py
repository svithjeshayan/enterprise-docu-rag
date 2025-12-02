import sqlite3
import os

db_path = "phase3_chatbot.db"

# Delete old DB if exists
if os.path.exists(db_path):
    os.remove(db_path)
    print("Old database deleted.")

# Create new DB with Phase 3 schema
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("""
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    content_hash TEXT,
    status TEXT DEFAULT 'active',
    added_at TIMESTAMP,
    updated_at TIMESTAMP
)
""")

conn.commit()
conn.close()

print("New Phase 3 database created successfully.")
