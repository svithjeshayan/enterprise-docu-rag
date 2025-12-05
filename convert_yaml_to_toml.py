#!/usr/bin/env python3
"""
Convert credentials.yaml to .streamlit/secrets.toml format
"""

import yaml
from pathlib import Path

def convert_yaml_to_toml():
    """Convert credentials.yaml to secrets.toml format"""
    
    # Read existing credentials.yaml
    with open('credentials.yaml', 'r') as f:
        creds = yaml.safe_load(f)
    
    # Create .streamlit directory if it doesn't exist
    streamlit_dir = Path('.streamlit')
    streamlit_dir.mkdir(exist_ok=True)
    
    # Build TOML content
    toml_lines = [
        "# Streamlit Secrets Configuration",
        "# Auto-generated from credentials.yaml",
        "",
        "# Add your OpenAI API key here:",
        'OPENAI_API_KEY = "sk-your-api-key-here"',
        "",
    ]
    
    # Convert usernames
    for username, user_data in creds['credentials']['usernames'].items():
        toml_lines.append(f"[authenticator.credentials.usernames.{username}]")
        toml_lines.append(f'email = "{user_data["email"]}"')
        toml_lines.append(f'name = "{user_data["name"]}"')
        toml_lines.append(f'password = "{user_data["password"]}"')
        toml_lines.append("")
    
    # Add cookie config
    cookie = creds['cookie']
    toml_lines.append("[authenticator.cookie]")
    toml_lines.append(f"expiry_days = {cookie['expiry_days']}")
    toml_lines.append(f'key = "{cookie["key"]}"')
    toml_lines.append(f'name = "{cookie["name"]}"')
    toml_lines.append("")
    
    # Add preauthorized (empty array)
    toml_lines.append("[authenticator.preauthorized]")
    toml_lines.append("# Empty array for preauthorized users")
    
    # Write to secrets.toml
    secrets_path = streamlit_dir / 'secrets.toml'
    with open(secrets_path, 'w') as f:
        f.write('\n'.join(toml_lines))
    
    print(f"✅ Created {secrets_path}")
    print("\n⚠️  IMPORTANT: Edit the file and add your real OpenAI API key!")
    print(f"   Location: {secrets_path.absolute()}")
    print("\n🔒 Make sure this file is in .gitignore")

if __name__ == "__main__":
    try:
        convert_yaml_to_toml()
    except FileNotFoundError:
        print("❌ Error: credentials.yaml not found!")
        print("   Make sure you're running this from your project directory")
    except Exception as e:
        print(f"❌ Error: {e}")