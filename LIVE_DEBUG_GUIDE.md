# 🔧 Live Debug Mode for RunPod

**Fast iteration debugging directly on RunPod without Docker rebuilds**

## 🚀 Quick Setup

### 1. Deploy Debug Version
```bash
# Sync to EC2 and build
rsync -avz --exclude='.git' /Users/antenehnegash/development/ml/sd3-large-docker/ ubuntu@ec2-35-173-103-83.compute-1.amazonaws.com:~/sd3-large-docker/

# Build and push debug version
ssh ubuntu@ec2-35-173-103-83.compute-1.amazonaws.com "cd ~/sd3-large-docker && docker build -t antenehmtk/sd3-docker-lazy-lora:debug-live . && docker push antenehmtk/sd3-docker-lazy-lora:debug-live"
```

### 2. Create RunPod with Debug Image
- Use image: `antenehmtk/sd3-docker-lazy-lora:debug-live`
- Container will automatically enter debug mode if server fails
- **Container stays alive forever** - no more shutdowns!

### 3. SSH into RunPod
```bash
# Use RunPod's SSH connection
ssh your-pod-connection-string
```

## 🎯 Debug Workflow

### Quick Commands Available:
```bash
/app/test_fix.sh         # Test the current train_lora fix
/app/restart_server.sh   # Restart just the API server  
/app/restart_celery.sh   # Restart just Celery worker
/app/show_logs.sh        # Show recent logs and status
```

### Typical Debug Session:
```bash
# 1. Test current state
/app/test_fix.sh

# 2. Edit the problematic file directly
nano /app/src/sd3_api/tasks/training_tasks.py

# 3. Apply changes (no Docker rebuild needed!)
/app/restart_server.sh

# 4. Test again
/app/test_fix.sh

# 5. Check logs if needed
/app/show_logs.sh
```

## 📁 Key Files You Can Edit Directly:

- `/app/src/sd3_api/tasks/training_tasks.py` - **Main file with train_lora**
- `/app/src/sd3_api/lora_api.py` - LoRA API endpoints
- `/app/src/sd3_api/api.py` - Main FastAPI app
- `/app/src/sd3_api/lora/trainer.py` - LoRA trainer logic

## 🔍 Current Issue We're Fixing

**Problem**: `train_lora` task stays in PENDING state forever
**Suspected cause**: Celery serialization issue with `Dict[str, Any]` type hints

**Working functions**: `test_celery()`, `test_training_imports()`
**Failing function**: `train_lora()`

## 🧪 Testing the Fix

The `test_fix.sh` script will:
1. ✅ Test basic Celery (should work)  
2. ✅ Test training imports (should work)
3. ✅ Create test child (should work)
4. ❌ Start train_lora (currently stays PENDING - this is what we're fixing)
5. 👀 Monitor task status (should show progress beyond PENDING after fix)

## 💡 Advantages of This Approach

- **No Docker rebuilds** - edit code directly on RunPod
- **Instant restarts** - restart just the API server, not the whole container
- **Always alive** - container never shuts down due to inactivity
- **Real-time testing** - see results immediately
- **Full SSH access** - complete debugging control

## 🔄 Keep-Alive Features

- Automatic heartbeat every 30 seconds
- Periodic self-pings to show activity  
- Continuous logging to prove activity
- Container stays alive forever in debug mode

This approach will let us iterate much faster on fixing the Celery serialization issue!