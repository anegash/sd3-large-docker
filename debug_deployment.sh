#!/bin/bash
# RunPod Deployment Debugging Script
# Run this on RunPod to diagnose why code changes aren't taking effect

echo "🔍 RUNPOD DEPLOYMENT DEBUGGING REPORT"
echo "======================================"
echo "Timestamp: $(date)"
echo "Hostname: $(hostname)"
echo

# 1. Environment & Location
echo "📁 ENVIRONMENT & LOCATION"
echo "-------------------------"
echo "Current directory: $(pwd)"
echo "User: $(whoami)"
echo "Python path: $(which python)"
echo "Poetry path: $(which poetry || echo 'Poetry not found')"
echo "Git branch: $(git branch --show-current 2>/dev/null || echo 'Not a git repo')"
echo

# 2. Git Status & Recent Changes
echo "📝 GIT STATUS & RECENT CHANGES"
echo "------------------------------"
echo "Git status:"
git status --porcelain || echo "Not a git repository"
echo
echo "Last 3 commits:"
git log --oneline -3 2>/dev/null || echo "No git history"
echo
echo "Uncommitted changes:"
git diff --name-only HEAD 2>/dev/null || echo "No git diff available"
echo

# 3. Verify Specific Fixes Are Present
echo "🔧 VERIFY FIXES IN CODE"
echo "----------------------"
echo "Checking for JSON serialization fix in lora_trainer.py:"
if [ -f "src/sd3_api/lora_trainer.py" ]; then
    grep -n "target_modules = list" src/sd3_api/lora_trainer.py || echo "❌ JSON fix NOT found"
    grep -n "pytorch_lora_weights.bin" src/sd3_api/lora_trainer.py || echo "❌ Weight file fix NOT found"
else
    echo "❌ lora_trainer.py not found"
fi
echo
echo "Checking for GPU offloading fix in pipeline.py:"
if [ -f "src/sd3_api/pipeline.py" ]; then
    grep -n "Use either CPU offloading OR manual GPU placement" src/sd3_api/pipeline.py || echo "❌ GPU offloading fix NOT found"
else
    echo "❌ pipeline.py not found"
fi
echo

# 4. Running Processes
echo "🏃 RUNNING PROCESSES"
echo "-------------------"
echo "All Python processes:"
ps aux | grep python | grep -v grep
echo
echo "Uvicorn processes:"
ps aux | grep uvicorn | grep -v grep
echo
echo "Processes using port 8000:"
lsof -i:8000 2>/dev/null || netstat -tulpn 2>/dev/null | grep :8000 || echo "No processes on port 8000"
echo

# 5. Server Process Details
echo "🖥️  SERVER PROCESS DETAILS"
echo "-------------------------"
MAIN_PID=$(pgrep -f "python.*main.py" | head -1)
if [ ! -z "$MAIN_PID" ]; then
    echo "Main server PID: $MAIN_PID"
    echo "Server working directory:"
    ls -la /proc/$MAIN_PID/cwd 2>/dev/null || echo "Cannot read process cwd"
    echo "Server command line:"
    cat /proc/$MAIN_PID/cmdline 2>/dev/null | tr '\0' ' ' || echo "Cannot read command line"
else
    echo "❌ No main.py process found"
fi
echo

# 6. Python Cache Files
echo "🗂️  PYTHON CACHE FILES"
echo "---------------------"
echo "Found .pyc files:"
find . -name "*.pyc" -type f | head -10
echo
echo "Found __pycache__ directories:"
find . -name "__pycache__" -type d | head -10
echo

# 7. Python Import Paths
echo "🐍 PYTHON IMPORT VERIFICATION"
echo "----------------------------"
echo "Python sys.path:"
python -c "import sys; print('\n'.join(sys.path))" 2>/dev/null || echo "Cannot run Python"
echo
echo "Module import test:"
if python -c "import src.sd3_api.pipeline" 2>/dev/null; then
    echo "✅ Can import src.sd3_api.pipeline"
    python -c "import src.sd3_api.pipeline; print('Pipeline module location:', src.sd3_api.pipeline.__file__)"
else
    echo "❌ Cannot import src.sd3_api.pipeline"
fi
echo
if python -c "import src.sd3_api.lora_trainer" 2>/dev/null; then
    echo "✅ Can import src.sd3_api.lora_trainer"
    python -c "import src.sd3_api.lora_trainer; print('LoRA trainer module location:', src.sd3_api.lora_trainer.__file__)"
else
    echo "❌ Cannot import src.sd3_api.lora_trainer"
fi
echo

# 8. File Permissions & Structure
echo "📋 FILE STRUCTURE & PERMISSIONS"
echo "-------------------------------"
echo "Project structure:"
ls -la | head -10
echo
echo "src/ directory:"
ls -la src/ 2>/dev/null | head -10
echo
echo "Key files permissions:"
[ -f "main.py" ] && ls -la main.py || echo "❌ main.py not found"
[ -f "src/sd3_api/pipeline.py" ] && ls -la src/sd3_api/pipeline.py || echo "❌ pipeline.py not found"
[ -f "src/sd3_api/lora_trainer.py" ] && ls -la src/sd3_api/lora_trainer.py || echo "❌ lora_trainer.py not found"
echo

# 9. Poetry Environment
echo "📦 POETRY ENVIRONMENT"
echo "--------------------"
if command -v poetry &> /dev/null; then
    echo "Poetry version: $(poetry --version)"
    echo "Poetry environment info:"
    poetry env info 2>/dev/null || echo "Cannot get poetry env info"
    echo
    echo "Virtual environment location:"
    poetry env list --full-path 2>/dev/null || echo "Cannot list poetry envs"
else
    echo "❌ Poetry not available"
fi
echo

# 10. Server Logs (last 50 lines)
echo "📜 RECENT SERVER LOGS"
echo "--------------------"
if [ -f "/workspace/logs/sd3_server.log" ]; then
    echo "Last 50 lines from server log:"
    tail -50 /workspace/logs/sd3_server.log
else
    echo "❌ Server log not found at /workspace/logs/sd3_server.log"
fi
echo

# 11. Multiple Code Copies Check
echo "🔍 DUPLICATE FILES CHECK"
echo "------------------------"
echo "Looking for multiple copies of key files:"
find /workspace -name "pipeline.py" -type f 2>/dev/null || echo "No pipeline.py files found"
find /workspace -name "lora_trainer.py" -type f 2>/dev/null || echo "No lora_trainer.py files found"
find /workspace -name "main.py" -type f 2>/dev/null || echo "No main.py files found"
echo

# 12. Disk Space & Memory
echo "💾 SYSTEM RESOURCES"
echo "------------------"
echo "Disk space:"
df -h . 2>/dev/null || echo "Cannot check disk space"
echo "Memory usage:"
free -h 2>/dev/null || echo "Cannot check memory"
echo

echo "🎯 DEBUGGING SUMMARY"
echo "===================="
echo "1. Check if fixes are present in code files above"
echo "2. Verify Python processes are running from correct directory" 
echo "3. Look for Python cache files that need clearing"
echo "4. Check if modules are importing from expected locations"
echo "5. Review server logs for startup errors or warnings"
echo
echo "Most common fix: Clear Python cache and restart"
echo "Commands: find . -name '*.pyc' -delete && find . -name '__pycache__' -exec rm -rf {} +"
echo "Then: ./stop_runpod.sh && ./start_runpod.sh"
echo
echo "Report complete at $(date)"