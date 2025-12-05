# simple_hash.py — Uses pure bcrypt (no streamlit-authenticator needed)
import bcrypt
import yaml

# Your users
users = {
    "john":   {"name": "John Doe",    "email": "john@company.com",   "password": "admin123"},
    "sarah":  {"name": "Sarah Lee",   "email": "sarah@company.com",  "password": "hr2025"},
    "guest":  {"name": "Guest User",  "email": "guest@company.com",  "password": "guest123"},
}

# Generate hashes
hashed_passwords = {}
for username, info in users.items():
    pwd = info["password"].encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(pwd, salt).decode('utf-8')
    hashed_passwords[username] = hashed

# Build config
final_config = {
    "credentials": {
        "usernames": {
            username: {
                "email": info["email"],
                "name": info["name"],
                "password": hashed
            }
            for username, info in users.items() for hashed in [hashed_passwords[username]]
        }
    },
    "cookie": {
        "expiry_days": 30,
        "key": "super-secret-random-key-2025-change-this-in-production",
        "name": "rag_chatbot_auth_cookie"
    },
    "preauthorized": []
}

print("\n=== COPY THIS INTO credentials.yaml ===\n")
print(yaml.dump(final_config, default_flow_style=False, sort_keys=False))
print("\n=== DONE! ===")