# RunPod Testing Sequence

## Quick Commands to Run on RunPod

```bash
# 1. Update code and fix deployment
cd /workspace/sdxl-api  # or wherever your code is
git pull
chmod +x fix_directory_references.sh
./fix_directory_references.sh

# 2. Verify everything is working
chmod +x verify_fixes.sh
./verify_fixes.sh

# 3. Test LoRA training (replace with your actual images)
curl -X POST "http://localhost:8000/train-lora" \
  -H "Content-Type: application/json" \
  -d '{"person_id": "test_person", "images": ["data:image/jpeg;base64,/9j/4AAQ..."]}'
```

## Expected Results After Fixes

✅ **Server logs clean** (no "Some layers...offloaded" warnings)  
✅ **LoRA training works** (no JSON serialization errors)  
✅ **Weight files created** (adapter_model.safetensors format)  
✅ **Server runs from correct directory** with latest fixes  

## Troubleshooting

If issues persist, run the debug script:
```bash
./debug_deployment.sh > debug_report.txt
cat debug_report.txt
```

The most common issue is Python module caching. The fix script handles this automatically.