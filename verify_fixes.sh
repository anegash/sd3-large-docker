#!/bin/bash
echo "🔍 VERIFYING ALL FIXES"
echo "====================="

# 1. Check server is running from correct directory
echo "1. Server directory check:"
SERVER_DIR=$(ps aux | grep -E "python.*main.py" | grep -v grep | awk '{for(i=11;i<=NF;i++) if($i ~ /main.py/) {print $(i-1)}' | head -1 | sed 's|/main.py||')
echo "   Server running from: $SERVER_DIR"

# 2. Check for GPU offloading warnings
echo "2. GPU offloading warnings:"
if grep -q "Some layers.*offloaded" /workspace/logs/sd3_server.log 2>/dev/null; then
    echo "   ❌ GPU offloading warnings still present"
else
    echo "   ✅ No GPU offloading warnings"
fi

# 3. Test LoRA endpoint availability
echo "3. LoRA endpoint test:"
if curl -s -f http://localhost:8000/train-lora > /dev/null; then
    echo "   ✅ LoRA endpoint accessible"
else
    echo "   ❌ LoRA endpoint not accessible"
fi

# 4. Check for our fixes in the code
echo "4. Code fixes verification:"
if python -c "import src.sd3_api.pipeline; import inspect; print('✅ GPU fix found' if 'Use either CPU offloading OR manual GPU placement' in inspect.getsource(src.sd3_api.pipeline.SD3Pipeline.load_base_model) else '❌ GPU fix missing')" 2>/dev/null; then
    echo "   GPU fix status checked"
else
    echo "   ❌ Could not verify GPU fix"
fi

if python -c "import src.sd3_api.lora_trainer; import inspect; print('✅ LoRA fix found' if 'adapter_model.safetensors' in inspect.getsource(src.sd3_api.lora_trainer.LoRATrainer.save_lora_weights) else '❌ LoRA fix missing')" 2>/dev/null; then
    echo "   LoRA fix status checked"
else
    echo "   ❌ Could not verify LoRA fix"
fi

echo
echo "✅ Verification complete!"