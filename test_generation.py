#!/usr/bin/env python3
"""
Test image generation and save the result.
"""

import requests
import base64
import json
from datetime import datetime

def test_generation():
    """Test image generation and save result."""
    
    print("🎨 Testing SD3.5 Large Image Generation")
    print("=" * 45)
    
    # API endpoint
    url = "http://localhost:8000/generate"
    
    # Generation request
    payload = {
        "prompt": "a happy young person smiling in natural lighting, portrait photo, professional photography",
        "steps": 15,  # Reduced steps for faster testing
        "guidance": 7.5
    }
    
    print(f"📝 Prompt: {payload['prompt']}")
    print(f"⚙️  Settings: {payload['steps']} steps, guidance {payload['guidance']}")
    print("\n🚀 Starting generation...")
    
    try:
        # Make request
        response = requests.post(url, json=payload, timeout=600)  # 10 minute timeout
        
        if response.status_code == 200:
            result = response.json()
            
            # Decode base64 image
            image_data = base64.b64decode(result["image"])
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_image_{timestamp}.png"
            
            with open(filename, "wb") as f:
                f.write(image_data)
            
            print(f"✅ Generation successful!")
            print(f"💾 Image saved as: {filename}")
            print(f"📊 Image size: {len(image_data)} bytes")
            
            return filename
            
        else:
            print(f"❌ Generation failed with status {response.status_code}")
            print(f"Error: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out - generation takes a while on Mac")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    result = test_generation()
    
    if result:
        print(f"\n🎉 Success! Check out your generated image: {result}")
        print("📱 You can open it with any image viewer or browser")
    else:
        print("\n💥 Generation failed. Check the server logs.")