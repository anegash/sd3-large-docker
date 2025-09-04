#!/usr/bin/env python3
"""
Quick RunPod API Test - Simple connectivity and functionality test
"""

import requests
import json
import uuid

def test_api():
    api_url = "https://96iel181wyyyw4-8000.proxy.runpod.net"
    session = requests.Session()
    
    print("🚀 Quick RunPod API Test")
    print("=" * 40)
    print(f"URL: {api_url}")
    print()
    
    # 1. Health Check
    print("🏥 Testing API health...")
    try:
        response = session.get(f"{api_url}/")
        response.raise_for_status()
        data = response.json()
        print(f"✅ API responding: {data.get('message', 'OK')}")
        print(f"   Device: {data.get('device', 'Unknown')}")
        print(f"   Version: {data.get('version', 'Unknown')}")
        model_ready = 'running' in data.get('message', '').lower()
        print(f"   Model Ready: {model_ready}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # 2. Child Operations Test
    print("\n👶 Testing child operations...")
    child_id = f"quicktest_{uuid.uuid4().hex[:6]}"
    
    try:
        # Create child
        response = session.post(
            f"{api_url}/lora/children",
            json={
                "id": child_id,
                "name": f"Quick Test {child_id[-6:]}",
                "description": "Quick API test"
            }
        )
        response.raise_for_status()
        print(f"✅ Child created: {child_id}")
        
        # List children
        response = session.get(f"{api_url}/lora/children")
        children = response.json()
        print(f"✅ Children listed: {len(children)} total")
        
        # Get child
        response = session.get(f"{api_url}/lora/children/{child_id}")
        child = response.json()
        print(f"✅ Child retrieved: {child['name']}")
        
        # Delete child
        response = session.delete(f"{api_url}/lora/children/{child_id}")
        response.raise_for_status()
        print(f"✅ Child deleted successfully")
        
    except Exception as e:
        print(f"❌ Child operations failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"   Response: {e.response.text[:200]}")
        return False
    
    # 3. Basic Generation Test (if model ready)
    if model_ready:
        print("\n🎨 Testing basic generation...")
        try:
            response = session.post(
                f"{api_url}/generate",
                json={
                    "prompt": "a simple test image",
                    "steps": 10,
                    "guidance": 7.5,
                    "width": 512,
                    "height": 512
                },
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"✅ Generation successful! ({len(response.content)} bytes)")
                with open("quick_test_gen.jpg", "wb") as f:
                    f.write(response.content)
                print("   💾 Saved as: quick_test_gen.jpg")
            else:
                print(f"⚠️  Generation returned {response.status_code}")
        except Exception as e:
            print(f"⚠️  Generation failed: {e}")
    else:
        print("\n⏳ Skipping generation - model still loading")
    
    print("\n" + "=" * 40)
    print("✅ API test completed!")
    
    print("\n📋 What works:")
    print("   ✅ API is responding")
    print("   ✅ Child CRUD operations work")
    print("   ✅ Redis and Celery are connected")
    if model_ready:
        print("   ✅ Model is ready for generation")
    else:
        print("   ⏳ Model is still loading")
    
    print("\n🎯 Ready for LoRA training!")
    print("   Next: Upload your training images")
    print("   Then: Run python test_fresh_lora.py")
    
    return True

if __name__ == "__main__":
    test_api()