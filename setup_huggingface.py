#!/usr/bin/env python3
"""HuggingFace Authentication Setup for SD3.5 Large"""

import sys

def setup_huggingface_auth():
    print("🤗 HuggingFace Authentication Setup")
    print("1. Get account: https://huggingface.co/join")
    print("2. Request access: https://huggingface.co/stabilityai/stable-diffusion-3.5-large")
    print("3. Create token: https://huggingface.co/settings/tokens")
    
    token = input("\n🔑 Enter your HuggingFace token: ").strip()
    if not token:
        print("❌ No token provided")
        return False
    
    try:
        from huggingface_hub import login, whoami
        login(token=token, add_to_git_credential=True)
        user_info = whoami()
        print(f"✅ Logged in as: {user_info['name']}")
        
        # Save to .env file
        with open(".env", "w") as f:
            f.write(f"HUGGINGFACE_TOKEN={token}\n")
        print("✅ Token saved to .env")
        return True
        
    except ImportError:
        print("❌ Run: poetry install")
        return False
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    if not setup_huggingface_auth():
        sys.exit(1)
    print("🎉 Setup complete! Run: poetry run python main.py")