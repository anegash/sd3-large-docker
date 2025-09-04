#!/usr/bin/env python3
"""
RunPod Setup Script for SD3.5 Large LoRA Training System

This script sets up a persistent environment on RunPod with all dependencies
installed in /workspace so they persist across pod restarts.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description="", check=True, shell=True):
    """Run a command and handle errors."""
    print(f"🔄 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=shell, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"   ✅ {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        if check:
            sys.exit(1)
        return e


def setup_workspace():
    """Set up workspace directories."""
    print("\n📁 Setting up workspace directories...")
    
    workspace_dirs = [
        "/workspace/lora_weights",
        "/workspace/models", 
        "/workspace/huggingface_cache",
        "/workspace/logs",
        "/workspace/venv"
    ]
    
    for dir_path in workspace_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {dir_path}")
    
    # Verify we're in the right project directory
    if not Path("/workspace/sd3-large-docker/pyproject.toml").exists():
        print("   ❌ Project not found at /workspace/sd3-large-docker")
        print("   📂 Please ensure you've cloned the repo to /workspace/sd3-large-docker")
        return False
    else:
        print("   ✅ Project found at /workspace/sd3-large-docker")
        return True


def install_system_dependencies():
    """Install system-level dependencies."""
    print("\n🔧 Installing system dependencies...")
    
    commands = [
        ("apt update", "Updating package lists"),
        ("apt install -y curl wget git build-essential", "Installing basic tools"),
        ("curl -sSL https://install.python-poetry.org | python3 -", "Installing Poetry")
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc)
    
    # Add poetry to PATH for this session
    os.environ["PATH"] = f"/root/.local/bin:{os.environ.get('PATH', '')}"


def setup_python_environment():
    """Set up Python environment with Poetry."""
    print("\n🐍 Setting up Python environment...")
    
    # Change to project directory (already exists at /workspace/sd3-large-docker)
    os.chdir("/workspace/sd3-large-docker")
    print(f"   📂 Working in: {os.getcwd()}")
    
    # Configure Poetry to use workspace venv
    commands = [
        ("poetry config virtualenvs.path /workspace/venv", "Configuring Poetry venv path"),
        ("poetry config virtualenvs.create true", "Enabling Poetry venv creation"),
        ("poetry config virtualenvs.in-project false", "Setting Poetry venv location"),
        ("poetry install --only main", "Installing main dependencies"),
        ("poetry install --with dev", "Installing development dependencies")
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc)


def setup_huggingface_auth():
    """Set up HuggingFace authentication."""
    print("\n🤗 Setting up HuggingFace authentication...")
    
    # Check if token is provided via environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    
    if not hf_token:
        print("   ⚠️  No HUGGINGFACE_TOKEN found in environment")
        print("   📝 Please set HUGGINGFACE_TOKEN in your RunPod environment variables")
        print("   🔗 Get your token from: https://huggingface.co/settings/tokens")
        return False
    
    # Save token to project .env file
    env_path = Path("/workspace/sd3-large-docker/.env")
    with open(env_path, "w") as f:
        f.write(f"HUGGINGFACE_TOKEN={hf_token}\n")
        f.write(f"WORKSPACE_DIR=/workspace\n")
        f.write(f"HF_HOME=/workspace/huggingface_cache\n")
    
    print(f"   ✅ HuggingFace token saved to {env_path}")
    return True


def setup_environment_variables():
    """Set up persistent environment variables."""
    print("\n🌍 Setting up environment variables...")
    
    # Create bashrc additions for persistent env vars
    bashrc_additions = """
# SD3 LoRA Training System Environment
export WORKSPACE_DIR=/workspace
export HF_HOME=/workspace/huggingface_cache
export TRANSFORMERS_CACHE=/workspace/huggingface_cache/transformers
export HF_DATASETS_CACHE=/workspace/huggingface_cache/datasets
export PATH="/root/.local/bin:$PATH"

# Activate Poetry environment
alias sd3-env="cd /workspace/sd3-large-docker && poetry shell"
alias sd3-run="cd /workspace/sd3-large-docker && poetry run python main.py"
alias sd3-status="cd /workspace/sd3-large-docker && poetry run python -c 'from src.sd3_api.pipeline import SD3Pipeline; p=SD3Pipeline(); print(p.status)'"
"""
    
    with open("/root/.bashrc", "a") as f:
        f.write(bashrc_additions)
    
    print("   ✅ Environment variables added to ~/.bashrc")


def create_startup_script():
    """Create startup script for the service."""
    print("\n🚀 Creating startup script...")
    
    startup_script = """#!/bin/bash
# SD3 LoRA Training System Startup Script

echo "🚀 Starting SD3 LoRA Training System..."

# Set environment variables
export WORKSPACE_DIR=/workspace
export HF_HOME=/workspace/huggingface_cache
export TRANSFORMERS_CACHE=/workspace/huggingface_cache/transformers
export HF_DATASETS_CACHE=/workspace/huggingface_cache/datasets
export PATH="/root/.local/bin:$PATH"

# Change to project directory
cd /workspace/sd3-large-docker

# Load environment variables from .env
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

echo "📋 Environment Status:"
echo "   Workspace: $WORKSPACE_DIR"
echo "   HuggingFace Cache: $HF_HOME"
echo "   Project Dir: $(pwd)"

# Start the server
echo "🔥 Starting FastAPI server..."
poetry run python main.py
"""
    
    script_path = Path("/workspace/start_sd3.sh")
    with open(script_path, "w") as f:
        f.write(startup_script)
    
    # Make executable
    script_path.chmod(0o755)
    
    print(f"   ✅ Startup script created: {script_path}")


def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    os.chdir("/workspace/sd3-large-docker")
    
    # Test Poetry environment
    result = run_command("poetry run python -c 'import torch; import diffusers; import peft; print(\"✅ All packages imported successfully\")'", 
                        "Testing package imports", check=False)
    
    if result.returncode != 0:
        print("   ❌ Package import test failed")
        return False
    
    # Test CUDA availability
    result = run_command("poetry run python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'",
                        "Testing CUDA availability", check=False)
    
    print("   ✅ Installation test completed")
    return True


def main():
    """Main setup function."""
    print("🎯 SD3.5 Large LoRA Training System - RunPod Setup")
    print("=" * 60)
    
    print("📋 This script will:")
    print("   • Set up persistent directories in /workspace")
    print("   • Install Poetry and Python dependencies") 
    print("   • Configure HuggingFace authentication")
    print("   • Create startup scripts")
    print("   • Set up environment variables")
    
    input("\n⏳ Press Enter to continue...")
    
    try:
        # Run setup steps
        if not setup_workspace():
            sys.exit(1)
        install_system_dependencies()
        setup_python_environment()
        
        if setup_huggingface_auth():
            print("   ✅ HuggingFace authentication configured")
        else:
            print("   ⚠️  HuggingFace authentication needs manual setup")
        
        setup_environment_variables()
        create_startup_script()
        
        if test_installation():
            print("\n🎉 Setup completed successfully!")
            print("\n📖 Next steps:")
            print("   1. Restart your terminal or run: source ~/.bashrc")
            print("   2. Start the server: /workspace/start_sd3.sh")
            print("   3. Or use alias: sd3-run")
            print("   4. API will be available at: http://localhost:8000")
            print("   5. Documentation: http://localhost:8000/docs")
            
            print("\n🔧 Useful commands:")
            print("   • sd3-env      : Activate Poetry environment")
            print("   • sd3-run      : Start the server") 
            print("   • sd3-status   : Check model status")
            
        else:
            print("\n⚠️  Setup completed with warnings. Check the logs above.")
            
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()