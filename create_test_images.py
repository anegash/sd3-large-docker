#!/usr/bin/env python3
"""
Create simple test images for LoRA training.
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_image(width=512, height=512, text="Test", color=None):
    """Create a simple colored image with text."""
    if color is None:
        color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
    
    # Create image
    img = Image.new('RGB', (width, height), color)
    
    # Add text
    draw = ImageDraw.Draw(img)
    try:
        # Try to use a larger font if available
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 40)
    except:
        font = ImageFont.load_default()
    
    # Get text size and center it
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    # Draw text
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return img

def main():
    """Create test training images."""
    print("🎨 Creating test training images for LoRA...")
    
    # Create directory
    output_dir = Path("data/children/aman/training_images")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create 10 test images with different colors and text
    prompts = [
        "aman smiling",
        "portrait of aman", 
        "aman happy",
        "photo of aman",
        "aman looking ahead",
        "aman outdoors",
        "aman casual",
        "aman indoors",
        "aman portrait",
        "aman face"
    ]
    
    colors = [
        (100, 150, 200),  # Light blue
        (150, 100, 200),  # Purple
        (200, 150, 100),  # Orange
        (100, 200, 150),  # Green
        (200, 100, 150),  # Pink
        (150, 200, 100),  # Lime
        (120, 120, 200),  # Blue
        (200, 120, 120),  # Red
        (120, 200, 120),  # Green
        (180, 180, 120),  # Yellow
    ]
    
    for i, (prompt, color) in enumerate(zip(prompts, colors)):
        # Create image
        img = create_test_image(512, 512, f"AMAN\n{i+1}", color)
        
        # Save as JPEG
        filename = f"IMG_{i+1:03d}.jpeg"
        filepath = output_dir / filename
        
        img.save(filepath, "JPEG", quality=95)
        print(f"✅ Created: {filepath}")
    
    print(f"\n🎉 Created {len(prompts)} test training images!")
    print(f"📁 Location: {output_dir}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("✅ Test images created successfully!")
        else:
            print("❌ Failed to create test images.")
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()