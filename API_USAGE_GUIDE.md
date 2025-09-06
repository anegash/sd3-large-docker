# SDXL LoRA Training API - Usage Guide

## Quick Start

### 1. Health Check
```bash
curl "https://your-server:8000/"
```

### 2. Upload Training Images (20+ recommended)
```bash
curl -X POST "https://your-server:8000/upload-images" \
  -F "person_id=your_person_id" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### 3. Train LoRA Model
```bash
curl -X POST "https://your-server:8000/train-lora" \
  -F "person_id=your_person_id" \
  -F "num_train_epochs=50" \
  -F "learning_rate=0.0001"
```

### 4. Generate Personalized Images
```bash
curl -X POST "https://your-server:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "professional photo of sks your_person_id",
    "steps": 25,
    "guidance": 7.5,
    "width": 1024,
    "height": 1024,
    "person_id": "your_person_id"
  }' | jq -r '.image' | base64 -d > output.png
```

## Detailed API Reference

### Training Endpoints

#### POST /upload-images
Upload training images for a person.

**Parameters:**
- `person_id` (form): Unique identifier for the person
- `files` (form): Multiple image files (JPEG/PNG)

**Example:**
```bash
curl -X POST "https://your-server:8000/upload-images" \
  -F "person_id=john_doe" \
  -F "files=@portrait1.jpg" \
  -F "files=@portrait2.jpg" \
  -F "files=@portrait3.jpg"
```

**Response:**
```json
{
  "message": "Successfully uploaded 3 images for john_doe",
  "person_id": "john_doe",
  "num_images": 3,
  "total_images": 3
}
```

#### POST /train-lora
Train a LoRA model for a person.

**Parameters:**
- `person_id` (form): Person to train for
- `num_train_epochs` (form): Number of training epochs (default: 100)
- `learning_rate` (form): Learning rate (default: 0.0001)
- `source_person_id` (form, optional): Copy images from existing person

**Example:**
```bash
curl -X POST "https://your-server:8000/train-lora" \
  -F "person_id=john_doe" \
  -F "num_train_epochs=75" \
  -F "learning_rate=0.0001"
```

**Response:**
```json
{
  "message": "LoRA training completed successfully for john_doe",
  "person_id": "john_doe"
}
```

### Generation Endpoints

#### GET /generate
Generate an image using query parameters.

**Parameters:**
- `prompt` (query): Text description of desired image
- `steps` (query): Number of inference steps (1-150, default: 20)
- `guidance` (query): Guidance scale (1.0-15.0, default: 7.5)
- `width` (query): Image width (512-2048, default: 1024)
- `height` (query): Image height (512-2048, default: 1024)
- `person_id` (query, optional): Person ID for LoRA weights

**Example:**
```bash
curl "https://your-server:8000/generate?prompt=portrait%20of%20sks%20john_doe&steps=25&guidance=7.5&person_id=john_doe"
```

#### POST /generate
Generate an image using JSON payload.

**Body:**
```json
{
  "prompt": "professional headshot of sks john_doe",
  "steps": 25,
  "guidance": 7.5,
  "width": 1024,
  "height": 1024,
  "person_id": "john_doe"
}
```

**Example:**
```bash
curl -X POST "https://your-server:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "artistic portrait of sks john_doe in studio lighting",
    "steps": 30,
    "guidance": 8.0,
    "width": 768,
    "height": 1024,
    "person_id": "john_doe"
  }'
```

**Response:**
```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAABA...base64_encoded_image"
}
```

### Management Endpoints

#### GET /lora
List all available LoRA models.

**Example:**
```bash
curl "https://your-server:8000/lora"
```

**Response:**
```json
{
  "person_ids": ["john_doe", "jane_smith", "bob_wilson"]
}
```

#### DELETE /lora/{person_id}
Delete a LoRA model and its weights.

**Example:**
```bash
curl -X DELETE "https://your-server:8000/lora/john_doe"
```

**Response:**
```json
{
  "message": "Successfully deleted LoRA model for john_doe"
}
```

#### GET /images
List all available image sets.

**Example:**
```bash
curl "https://your-server:8000/images"
```

**Response:**
```json
{
  "image_sets": [
    {
      "person_id": "john_doe",
      "num_images": 15,
      "images_ready": true
    },
    {
      "person_id": "jane_smith", 
      "num_images": 8,
      "images_ready": false
    }
  ]
}
```

#### GET /images/{person_id}
Get image status for a specific person.

**Example:**
```bash
curl "https://your-server:8000/images/john_doe"
```

**Response:**
```json
{
  "person_id": "john_doe",
  "num_images": 15,
  "images_ready": true
}
```

#### DELETE /images/{person_id}
Delete all images for a specific person.

**Example:**
```bash
curl -X DELETE "https://your-server:8000/images/john_doe"
```

**Response:**
```json
{
  "message": "Successfully deleted images for john_doe"
}
```

## Best Practices

### Training Data Guidelines

1. **Quantity**: Upload 15-30 high-quality images per person
2. **Variety**: Include different angles, expressions, and lighting
3. **Quality**: Use high-resolution images (1024px+ recommended)
4. **Consistency**: Keep the same person across all images
5. **Format**: JPEG or PNG formats supported

### Prompt Engineering

1. **Always use the unique token**: Include `sks person_id` in your prompts
2. **Be specific**: "professional headshot of sks john_doe" works better than "photo of person"
3. **Style descriptors**: Add style terms like "professional", "artistic", "candid"
4. **Composition**: Specify composition like "close-up", "full body", "portrait"

**Good Prompts:**
- `"professional headshot of sks john_doe in business attire"`
- `"artistic portrait of sks jane_smith with dramatic lighting"`
- `"casual photo of sks bob_wilson smiling outdoors"`

**Avoid:**
- Generic prompts without the unique token
- Overly complex or contradictory descriptions
- Inappropriate or harmful content requests

### Training Parameters

| Parameter | Recommended | Range | Notes |
|-----------|-------------|-------|-------|
| `num_train_epochs` | 50-100 | 10-500 | More epochs = better personalization, longer training |
| `learning_rate` | 0.0001 | 1e-5 to 1e-3 | Lower = more stable, higher = faster convergence |

### Generation Parameters

| Parameter | Recommended | Range | Notes |
|-----------|-------------|-------|-------|
| `steps` | 25-30 | 10-50 | More steps = better quality, slower generation |
| `guidance` | 7.5-8.5 | 1.0-15.0 | Higher = more prompt adherence, less creativity |
| `width/height` | 1024x1024 | 512-2048 | Must be divisible by 8, 1024x1024 is native SDXL |

## Error Handling

### Common Errors

#### 400 Bad Request
- Missing required parameters
- Invalid image format
- No images provided for training

#### 404 Not Found
- Person ID doesn't exist
- LoRA model not found

#### 500 Internal Server Error
- Training failed (check logs)
- Model loading error
- Generation failed

#### 503 Service Unavailable
- Model still loading
- Training in progress

### Troubleshooting

1. **Training fails**: Ensure you have uploaded at least 5 images
2. **No personalization**: Check that you're using the correct `person_id` and `sks` token
3. **Poor quality**: Try increasing steps or adjusting guidance scale
4. **Memory errors**: Reduce image resolution or batch size

## Integration Examples

### Python
```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Generate image
response = requests.post("https://your-server:8000/generate", json={
    "prompt": "professional photo of sks john_doe",
    "steps": 25,
    "guidance": 7.5,
    "person_id": "john_doe"
})

# Save image
image_data = base64.b64decode(response.json()["image"])
image = Image.open(BytesIO(image_data))
image.save("generated.png")
```

### JavaScript/Node.js
```javascript
const fs = require('fs');

async function generateImage() {
  const response = await fetch('https://your-server:8000/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: 'portrait of sks john_doe',
      steps: 25,
      person_id: 'john_doe'
    })
  });
  
  const data = await response.json();
  const buffer = Buffer.from(data.image, 'base64');
  fs.writeFileSync('generated.png', buffer);
}
```

### cURL with Image Saving
```bash
# Generate and save image
curl -X POST "https://your-server:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "headshot of sks john_doe", "person_id": "john_doe"}' \
  | jq -r '.image' \
  | base64 -d > generated_image.png

# Verify image was created
file generated_image.png
```

## Performance Tips

1. **Batch Operations**: Upload multiple images at once rather than one-by-one
2. **Optimal Resolution**: Use 1024x1024 for best quality/speed balance
3. **Reasonable Steps**: 25-30 steps provide good quality without excessive time
4. **Monitor Training**: Training typically takes 20-40 minutes depending on epochs
5. **Cache Models**: Trained models persist between server restarts

## Version Information

The API includes version tracking for debugging and deployment verification:

```bash
curl "https://your-server:8000/" | jq
```

```json
{
  "message": "Stable Diffusion XL API is running!",
  "device": "CUDA GPU: NVIDIA A40",
  "version": "1.1.1",
  "branch": "feature/sdxl-migration",
  "commit": "9510382"
}
```

This helps ensure you're running the expected version after deployments.