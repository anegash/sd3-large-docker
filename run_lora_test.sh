#!/bin/bash

# SD3.5 Large LoRA Training Test Runner
# This script helps run the complete LoRA training test workflow

set -e  # Exit on any error

echo "🧪 SD3.5 Large LoRA Training Test Runner"
echo "========================================"

# Check if we're on RunPod (look for typical RunPod paths)
if [[ -d "/workspace" && -d "/workspace/sd3-large-docker" ]]; then
    echo "🔧 RunPod environment detected"
    IMAGES_DIR="/workspace/training_data"
    API_URL="http://localhost:8000"
    
    # Copy test images to workspace if they don't exist
    if [[ ! -d "$IMAGES_DIR" ]]; then
        mkdir -p "$IMAGES_DIR"
        echo "📁 Created training data directory: $IMAGES_DIR"
        echo "   Please upload your training images to this directory"
        echo "   Or modify the script to point to your image location"
    fi
    
    cd /workspace/sd3-large-docker
else
    echo "🖥️  Local development environment detected"
    IMAGES_DIR="/Users/antenehnegash/Downloads/aman"
    API_URL="http://localhost:8000"
fi

# Check if API server is running
echo "🔍 Checking if API server is running..."
if curl -s "$API_URL/" > /dev/null 2>&1; then
    echo "✅ API server is running at $API_URL"
else
    echo "❌ API server is not running at $API_URL"
    echo ""
    echo "Please start the server first:"
    echo "  RunPod:  ./start_app.sh"
    echo "  Local:   poetry run python main.py"
    echo ""
    exit 1
fi

# Check if Celery worker is running
echo "🔍 Checking if Celery worker is running..."
if pgrep -f "celery.*worker" > /dev/null; then
    echo "✅ Celery worker is running"
else
    echo "⚠️  Celery worker not detected"
    echo ""
    echo "Please start the Celery worker:"
    echo "  RunPod:  ./start_worker.sh"
    echo "  Local:   poetry run celery -A src.sd3_api.tasks.celery_app worker --loglevel=info -Q training"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if training images exist
echo "🔍 Checking training images..."
if [[ -d "$IMAGES_DIR" ]]; then
    IMAGE_COUNT=$(find "$IMAGES_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
    echo "   Found $IMAGE_COUNT training images in $IMAGES_DIR"
    
    if [[ $IMAGE_COUNT -lt 3 ]]; then
        echo "⚠️  Warning: Only $IMAGE_COUNT images found. Recommend at least 10-20 for good LoRA training."
    fi
else
    echo "❌ Training images directory not found: $IMAGES_DIR"
    exit 1
fi

# Ask for test mode
echo ""
echo "🚀 Ready to run LoRA training test!"
echo "Choose test mode:"
echo "  1) Quick test (100 steps, ~5 minutes)"
echo "  2) Full test (500 steps, ~20 minutes)"
echo "  3) Custom parameters"
echo ""
read -p "Select mode (1-3): " -n 1 -r MODE
echo ""

# Prepare arguments
TEST_ARGS="--api-url $API_URL --images-dir $IMAGES_DIR"

case $MODE in
    1)
        echo "🏃‍♂️ Running quick test mode..."
        TEST_ARGS="$TEST_ARGS --quick"
        ;;
    2)
        echo "🐢 Running full test mode..."
        ;;
    3)
        echo "⚙️  Custom mode selected"
        read -p "Child ID (default: aman): " CHILD_ID
        CHILD_ID=${CHILD_ID:-aman}
        TEST_ARGS="$TEST_ARGS --child-id $CHILD_ID"
        
        read -p "Run quick test? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            TEST_ARGS="$TEST_ARGS --quick"
        fi
        ;;
    *)
        echo "❌ Invalid selection"
        exit 1
        ;;
esac

# Create logs directory
mkdir -p logs

# Run the test
echo ""
echo "🧪 Starting LoRA training test..."
echo "   Command: python test_lora_training.py $TEST_ARGS"
echo "   Logs will be saved to: lora_training_test.log"
echo ""

# Check if we're using Poetry
if command -v poetry > /dev/null && [[ -f "pyproject.toml" ]]; then
    poetry run python test_lora_training.py $TEST_ARGS
else
    python test_lora_training.py $TEST_ARGS
fi

TEST_EXIT_CODE=$?

echo ""
echo "========================================"
if [[ $TEST_EXIT_CODE -eq 0 ]]; then
    echo "🎉 LoRA training test COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Check the generated images:"
    ls -la generated_test_*.jpg 2>/dev/null || echo "   No generated images found"
else
    echo "❌ LoRA training test FAILED"
    echo ""
    echo "Check the logs for details:"
    echo "   tail -f lora_training_test.log"
fi

echo ""
echo "📊 Useful commands for debugging:"
echo "   View test logs:     tail -f lora_training_test.log"
echo "   Check API health:   curl $API_URL/"
echo "   Check processes:    ps aux | grep -E '(uvicorn|celery)'"
echo "   Monitor GPU:        nvidia-smi"
echo "   View generated:     ls -la generated_test_*.jpg"

exit $TEST_EXIT_CODE