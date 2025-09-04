# LoRA Training Steps on RunPod

**API Endpoint**: https://ckqn9ap916vnnt-8000.proxy.runpod.net/

This guide provides step-by-step instructions for testing LoRA training on the deployed RunPod instance.

## 📋 Prerequisites

- RunPod instance running with latest Docker image
- Training images ready (at least 5 images in JPG/PNG format)
- API endpoint accessible: https://ckqn9ap916vnnt-8000.proxy.runpod.net/

## 🔄 Deployment Workflow

**Important:** Since we're on Mac (can't build Docker images locally) and EC2 has no git, follow this deployment sequence:

1. **Local Development:** Make changes in local repository
2. **Transfer Files:** Copy updated files to EC2 instance:
   ```bash
   # Sync all files to EC2 (excluding .git)
   rsync -avz --exclude='.git' /Users/antenehnegash/development/ml/sd3-large-docker/ ubuntu@ec2-35-173-103-83.compute-1.amazonaws.com:/home/ubuntu/sd3-large-docker/
   ```
3. **Build & Push on EC2:** Build and push Docker image:
   ```bash
   ssh ubuntu@ec2-35-173-103-83.compute-1.amazonaws.com "cd /home/ubuntu/sd3-large-docker && docker build -t antenehmtk/sd3-docker-lazy-lora:latest . && docker push antenehmtk/sd3-docker-lazy-lora:latest"
   ```
4. **Restart RunPod Pod:** Stop and start the Pod in RunPod console to pull new image
5. **Verify Version:** Check that new version is deployed:
   ```bash
   # Check basic version in health endpoint
   curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/"
   
   # Check detailed build information
   curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/version"
   ```
6. **Test Functionality:** Verify deployment by testing endpoints since local testing isn't possible on Mac

**Note:** We use EC2 as our Docker build environment because Mac can't build the required Linux containers, and RunPod automatically pulls the latest image when restarted.

## 🔧 Step 0: Verify System Health

```bash
# Check API health and GPU status
curl https://ckqn9ap916vnnt-8000.proxy.runpod.net/

# Test Celery worker functionality
curl -X POST https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/debug/test-celery

# Check Celery worker information
curl https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/debug/celery-info
```

## 🎯 Step 1: Create a New Child

Create a unique child ID for your training session:

```bash
curl -X POST "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/children" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "test_child_001", 
    "name": "Test Child",
    "description": "Child for LoRA training testing"
  }'
```

**Expected Response:**
```json
{
  "id": "test_child_001",
  "name": "Test Child",
  "description": "Child for LoRA training testing",
  "created_at": "2025-09-03T...",
  "updated_at": "2025-09-03T...",
  "training_image_count": 0,
  "has_active_model": false
}
```

## 📸 Step 2: Upload Training Images

Upload all training images at once (minimum 5, maximum recommended 30):

```bash
# Upload all 26 images from the aman directory
cd /Users/antenehnegash/Downloads/aman/
curl -X POST "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/children/test_child_001/images" \
  -F "files=@IMG_6557.jpeg" \
  -F "files=@IMG_6558.jpeg" \
  -F "files=@IMG_6559.jpeg" \
  -F "files=@IMG_6560.jpeg" \
  -F "files=@IMG_6561.jpeg" \
  -F "files=@IMG_6562.jpeg" \
  -F "files=@IMG_6563.jpeg" \
  -F "files=@IMG_6564.jpeg" \
  -F "files=@IMG_6565.jpeg" \
  -F "files=@IMG_6566.jpeg" \
  -F "files=@IMG_6567.jpeg" \
  -F "files=@IMG_6568.jpeg" \
  -F "files=@IMG_6569.jpeg" \
  -F "files=@IMG_6570.jpeg" \
  -F "files=@IMG_6571.jpeg" \
  -F "files=@IMG_6572.jpeg" \
  -F "files=@IMG_6573.jpeg" \
  -F "files=@IMG_6574.jpeg" \
  -F "files=@IMG_6575.jpeg" \
  -F "files=@IMG_6576.jpeg" \
  -F "files=@IMG_6577.jpeg" \
  -F "files=@IMG_6578.jpeg" \
  -F "files=@IMG_6579.jpeg" \
  -F "files=@IMG_6580.jpeg" \
  -F "files=@IMG_6581.jpeg" \
  -F "files=@IMG_6582.jpeg"

# Alternative: Upload just the minimum 5 for quick testing
curl -X POST "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/children/test_child_001/images" \
  -F "files=@IMG_6557.jpeg" \
  -F "files=@IMG_6558.jpeg" \
  -F "files=@IMG_6559.jpeg" \
  -F "files=@IMG_6560.jpeg" \
  -F "files=@IMG_6561.jpeg"
```

**Expected Response:**
```json
{
  "message": "5 images uploaded successfully",
  "uploaded_images": [...],
  "total_images": 5
}
```

## 🚀 Step 3: Start LoRA Training

Initiate the training process with custom configuration:

```bash
curl -X POST "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/children/test_child_001/train" \
  -H "Content-Type: application/json" \
  -d '{
    "training_config": {
      "training_steps": 100,
      "learning_rate": 1e-4,
      "lora_rank": 64,
      "batch_size": 1,
      "gradient_accumulation_steps": 4
    }
  }'
```

**Expected Response:**
```json
{
  "message": "Training started successfully",
  "child_id": "test_child_001",
  "model_id": 123456,
  "task_id": "abc-123-def-456",
  "status": "pending"
}
```

## 📊 Step 4: Monitor Training Progress

Check the training status periodically:

```bash
# Check training status
curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/children/test_child_001/training-status"
```

**Response During Training:**
```json
{
  "model_id": 123456,
  "child_id": "test_child_001",
  "status": "training",
  "progress": 0.45,
  "message": "Step 45/100, Loss: 0.0234",
  "loss": 0.0234,
  "error": null,
  "task_id": "abc-123-def-456",
  "started_at": "2025-09-03T...",
  "estimated_completion": "2025-09-03T..."
}
```

**Response When Completed:**
```json
{
  "status": "completed",
  "progress": 1.0,
  "message": "Training completed successfully"
}
```

**Response If Failed:**
```json
{
  "status": "failed",
  "progress": 0.0,
  "error": "Error message here"
}
```

## 🎨 Step 5: Generate Images with Trained LoRA

Once training is complete, generate images using the trained model:

```bash
curl -X POST "https://ckqn9ap916vnnt-8000.proxy.runpod.net/generate/with-children" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a happy {test_child_001} playing in the park",
    "steps": 20,
    "guidance": 7.5
  }'
```

The `{test_child_001}` token will be replaced with the trained LoRA representation.

## 🐛 Debugging Training Failures

**Status**: As of v0.6.0, we have **isolated the root cause** to the `train_lora` function specifically. All infrastructure is confirmed working.

### **Systematic Debugging Workflow:**

#### 1. Check System Health
```bash
# Verify version and deployment
curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/version"

# Check basic connectivity
curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/"
```

#### 2. Test Celery Infrastructure  
```bash
# Test basic Celery functionality
curl -X POST "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/debug/test-celery"

# Check result (should complete in ~5 seconds)
curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/debug/task-status/{TASK_ID}"
```

#### 3. Test Training Dependencies
```bash
# Test all training imports and dependencies
curl -X POST "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/debug/test-training-imports"

# Check result (should show successful imports)
curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/debug/task-status/{TASK_ID}"
```

#### 4. Check Celery Worker Configuration
```bash  
# View worker status and registered tasks
curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/debug/celery-info"
```

#### 5. Monitor Training Task Status
```bash
# Check specific training task (using task_id from training response)
curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/debug/task-status/{TASK_ID}"

# Check enhanced training status (shows real progress, not fake 50%)
curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/children/{CHILD_ID}/training-status"
```

### **Known Working Components (v0.6.0):**
- ✅ **Celery Worker**: Executes tasks successfully
- ✅ **All Dependencies**: PyTorch, LoRATrainer, storage_manager available
- ✅ **Task Registration**: `train_lora` properly registered
- ✅ **Queue Routing**: Tasks routed to correct queues
- ✅ **Progress Tracking**: Real-time status (fixed fake 50% issue)

### **Isolated Issue:**
- ❌ **Only `train_lora` task stays in PENDING** - all other tasks execute successfully
- Root cause isolated to something specific within the `train_lora` function

### **Additional Debug Endpoints:**
```bash
# View child details
curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/children/{CHILD_ID}"

# List training images
curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/children/{CHILD_ID}/images"

# System statistics
curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/system/statistics"
```

## 📝 Training Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training_steps` | 1000 | Total number of training steps |
| `learning_rate` | 1e-4 | Learning rate for optimization |
| `lora_rank` | 64 | LoRA rank (higher = more parameters) |
| `batch_size` | 1 | Training batch size |
| `gradient_accumulation_steps` | 4 | Steps to accumulate gradients |
| `save_steps` | 250 | Save checkpoint every N steps |
| `validation_steps` | 100 | Validate model every N steps |
| `mixed_precision` | "fp16" | Use mixed precision training |

## 🚨 Common Issues and Solutions

### Issue: Training Status Shows "failed" Without Error
**Solution:** Check Celery worker is running:
```bash
curl "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/debug/celery-info"
```

### Issue: Training Task Stuck in "PENDING" State
**Solution:** This typically occurs when code changes have been pushed but RunPod Pod hasn't restarted to pull the new Docker image:
1. **Verify latest code is deployed:** Check that your changes have been pushed to GitHub and the Docker image updated on Docker Hub
2. **Restart RunPod Pod:** Stop and start the Pod in RunPod console to pull the latest image
3. **Test Celery functionality:** Use the test endpoint to verify worker is functioning:
   ```bash
   curl -X POST "https://ckqn9ap916vnnt-8000.proxy.runpod.net/lora/debug/test-celery"
   ```

### Issue: "Insufficient training images"
**Solution:** Ensure at least 5 images are uploaded before training.

### Issue: Out of Memory
**Solution:** Reduce `batch_size` to 1 or reduce `lora_rank` to 32.

### Issue: Training Stuck at 0%
**Solution:** Check GPU availability and restart Pod if necessary.

### Issue: Training Steps Validation Error
**Solution:** Minimum training steps is 100. Use at least 100 steps in training configuration.

## 🔄 Complete Test Flow Script

Here's a complete script to test the entire flow:

```bash
#!/bin/bash

API_URL="https://ckqn9ap916vnnt-8000.proxy.runpod.net"
CHILD_ID="test_$(date +%s)"  # Unique ID with timestamp

echo "🔧 Testing LoRA Training System"
echo "================================"

# Step 0: Health Check
echo "📋 Checking system health..."
curl -s "$API_URL/" | jq .

# Step 1: Create Child
echo "👶 Creating child: $CHILD_ID"
curl -s -X POST "$API_URL/lora/children" \
  -H "Content-Type: application/json" \
  -d "{\"id\": \"$CHILD_ID\", \"name\": \"Test Child\", \"description\": \"Automated test\"}" | jq .

# Step 2: Upload Images (requires local images)
echo "📸 Uploading training images..."
# Update these paths to your actual image files
curl -s -X POST "$API_URL/lora/children/$CHILD_ID/images" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "files=@image4.jpg" \
  -F "files=@image5.jpg" | jq .

# Step 3: Start Training
echo "🚀 Starting training..."
RESPONSE=$(curl -s -X POST "$API_URL/lora/children/$CHILD_ID/train" \
  -H "Content-Type: application/json" \
  -d '{"training_config": {"training_steps": 50, "lora_rank": 32}}')
echo "$RESPONSE" | jq .

# Step 4: Monitor Progress
echo "📊 Monitoring progress..."
for i in {1..10}; do
  sleep 10
  STATUS=$(curl -s "$API_URL/lora/children/$CHILD_ID/training-status")
  echo "Progress check $i/10:"
  echo "$STATUS" | jq '.status, .progress, .message'
  
  # Check if completed or failed
  if echo "$STATUS" | jq -e '.status == "completed" or .status == "failed"' > /dev/null; then
    break
  fi
done

# Step 5: Final Status
echo "✅ Final training status:"
curl -s "$API_URL/lora/children/$CHILD_ID/training-status" | jq .
```

## 📚 Additional Resources

- **API Documentation**: https://ckqn9ap916vnnt-8000.proxy.runpod.net/docs
- **View Logs**: Check RunPod console for detailed logs
- **SSH Access**: Connect to Pod for direct troubleshooting

## 💡 Tips for Successful Training

1. **Image Quality**: Use high-quality, clear images of the subject
2. **Image Variety**: Include different angles, expressions, and backgrounds
3. **Consistent Subject**: Ensure the same subject appears in all images
4. **Training Steps**: Start with 100-200 steps for testing, increase for better quality
5. **Monitor Progress**: Check status every 30 seconds to catch any issues early

Remember to replace `test_child_001` with unique IDs for each training session to avoid conflicts!