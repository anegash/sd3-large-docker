#!/usr/bin/env python3
"""RunPod Setup Script for SD3.5 Large LoRA Training System"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description="", check=True):
    """Run a command and handle errors."""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if check:
            sys.exit(1)
        return e


def setup_workspace():
    """Set up workspace directories."""
    print("📁 Setting up workspace directories...")
    workspace_dirs = ["/workspace/lora_weights", "/workspace/huggingface_cache", "/workspace/logs"]
    
    for dir_path in workspace_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    if not Path("/workspace/sd3-large-docker/pyproject.toml").exists():
        print("❌ Project not found at /workspace/sd3-large-docker")
        return False
    return True

def install_dependencies():
    """Install system dependencies and Poetry environment."""
    print("🔧 Installing dependencies...")
    run_command("apt update && apt install -y curl git", "Installing system packages")
    run_command("curl -sSL https://install.python-poetry.org | python3 -", "Installing Poetry")
    
    os.environ["PATH"] = f"/root/.local/bin:{os.environ.get('PATH', '')}"
    os.chdir("/workspace/sd3-large-docker")
    
    run_command("poetry config virtualenvs.path /workspace/venv", "Configuring Poetry")
    run_command("poetry install --only main", "Installing Python dependencies")

def setup_huggingface_auth():
    """Set up HuggingFace authentication."""
    print("🤗 Setting up HuggingFace authentication...")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        print("⚠️  Set HF_TOKEN in RunPod environment variables")
        return False
    
    env_path = Path("/workspace/sd3-large-docker/.env")
    with open(env_path, "w") as f:
        f.write(f"HUGGINGFACE_TOKEN={hf_token}\n")
        f.write(f"HF_HOME=/workspace/huggingface_cache\n")
    return True

def create_startup_script():
    """Create startup script."""
    print("🚀 Creating startup script...")
    script = """#!/bin/bash
export HF_HOME=/workspace/huggingface_cache
export PATH="/root/.local/bin:$PATH"
cd /workspace/sd3-large-docker
[ -f .env ] && export $(cat .env | xargs)
poetry run python main.py
"""
    
    with open("/workspace/start_sd3.sh", "w") as f:
        f.write(script)
    Path("/workspace/start_sd3.sh").chmod(0o755)

def main():
    """Main setup function."""
    print("🎯 SD3.5 Large - RunPod Setup")
    
    if not setup_workspace():
        sys.exit(1)
    install_dependencies()
    setup_huggingface_auth()
    create_startup_script()
    
    print("🎉 Setup complete!")
    print("Start server: /workspace/start_sd3.sh")


if __name__ == "__main__":
    main()