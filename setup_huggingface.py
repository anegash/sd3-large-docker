#!/usr/bin/env python3
"""
HuggingFace Authentication Setup Script for SD3.5 Large

This script helps you set up authentication to access the SD3.5 Large model.
"""

import os
import sys
from pathlib import Path

def setup_huggingface_auth():
    """Set up HuggingFace authentication."""
    
    print("🤗 Setting up HuggingFace Authentication for SD3.5 Large")
    print("=" * 60)
    
    print("\n📋 Steps to get access:")
    print("1. Create a HuggingFace account at: https://huggingface.co/join")
    print("2. Request access to SD3.5 Large at: https://huggingface.co/stabilityai/stable-diffusion-3.5-large")
    print("3. Create an access token at: https://huggingface.co/settings/tokens")
    print("   - Choose 'Read' permissions")
    print("   - Name it something like 'SD3.5-API-Token'")
    
    print("\n🔑 Enter your HuggingFace token:")
    token = input("Token: ").strip()
    
    if not token:
        print("❌ No token provided. Exiting.")
        return False
    
    # Try to login using huggingface_hub
    try:
        from huggingface_hub import login, whoami
        
        # Login with the token
        login(token=token, add_to_git_credential=True)
        
        # Verify login worked
        user_info = whoami()
        print(f"✅ Successfully logged in as: {user_info['name']}")
        
        # Also save to environment file for Docker
        env_path = Path(".env")
        with open(env_path, "w") as f:
            f.write(f"HUGGINGFACE_TOKEN={token}\n")
        print(f"✅ Token saved to {env_path}")
        
        print("\n🎉 HuggingFace authentication setup complete!")
        print("\nNext steps:")
        print("1. Make sure you have access to SD3.5 Large model")
        print("2. Run: poetry run python main.py")
        
        return True
        
    except ImportError:
        print("❌ huggingface_hub not installed. Run: poetry install")
        return False
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        print("\nTroubleshooting:")
        print("- Make sure your token is correct")
        print("- Ensure you have access to the SD3.5 Large model")
        print("- Try creating a new token with 'Read' permissions")
        return False

if __name__ == "__main__":
    if not setup_huggingface_auth():
        sys.exit(1)
    print("\n✅ You're all set! Try running the server now.")