import os
import sqlite3

# Print working directory
print("Current working directory:", os.getcwd())

# List all files in project folder
print("\nFiles in folder:")
for f in os.listdir("."):
    print("-", f)

# Try opening DB
try:
    conn = sqlite3.connect("phase3_chatbot.db")
    c = conn.cursor()
    c.execute("PRAGMA table_info(documents)")
    print("\nDB Columns:")
    for col in c.fetchall():
        print(col)
    conn.close()
except Exception as e:
    print("Error reading DB:", e)
