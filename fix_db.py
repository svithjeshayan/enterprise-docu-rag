import sqlite3

conn = sqlite3.connect("phase3_chatbot.db")
c = conn.cursor()
try:
    c.execute("ALTER TABLE documents ADD COLUMN chunk_count INTEGER")
    conn.commit()
    print("Column 'chunk_count' added successfully.")
except sqlite3.OperationalError as e:
    print(f"Error: {e} - Column may already exist or table is missing.")
finally:
    conn.close()