import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SamProcessor, SamModel
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load SAM model & processor
checkpoint = "facebook/sam-vit-large"
processor = SamProcessor.from_pretrained(checkpoint)
model = SamModel.from_pretrained(checkpoint).to(device).eval()

# Load image
image_path = "test4.png"  # Change to your image path
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

image = Image.open(image_path)
if image.mode != "RGB":
    image = image.convert("RGB")

print(f"Loaded image: {image_path}")
print(f"Image size: {image.size}")

# Run inference & time it
start_time = time.time()
with torch.no_grad():
    # In SAM, we need prompts to generate masks; we'll simulate a center point prompt
    input_points = [[[image.width // 2, image.height // 2]]]  # one point in the middle
    inputs_with_prompt = processor(
        images=image,
        input_points=input_points,
        return_tensors="pt"
    ).to(device)
    
    outputs = model(**inputs_with_prompt)
    masks = outputs.pred_masks  # shape: [batch, num_masks, H, W]

inference_time = time.time() - start_time
print(f"Inference completed in {inference_time:.2f} seconds")

# Postprocess mask - Fix the dimension handling
print(f"Original masks shape: {masks.shape}")

# SAM returns shape [batch, num_points, num_masks, H, W]
# Take the first batch, first point, and first mask
mask = masks[0, 0, 0].cpu().numpy()  # shape: [H, W]
binary_mask = (mask > 0).astype(np.uint8)  # convert to binary

print(f"Mask shape after processing: {binary_mask.shape}")
print(f"Unique values in mask: {np.unique(binary_mask)}")

# Resize mask to match original image size if needed
if binary_mask.shape != (image.height, image.width):
    from scipy.ndimage import zoom
    # Calculate zoom factors
    zoom_h = image.height / binary_mask.shape[0]
    zoom_w = image.width / binary_mask.shape[1]
    binary_mask = zoom(binary_mask, (zoom_h, zoom_w), order=0).astype(np.uint8)
    print(f"Resized mask shape: {binary_mask.shape}")

# Create color mask with correct dimensions
color_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 3), dtype=np.uint8)
color_mask[binary_mask == 1] = [255, 0, 0]  # red for segmentation

# Create overlay for better visualization
image_array = np.array(image)
overlay = image_array.copy()
overlay[binary_mask == 1] = overlay[binary_mask == 1] * 0.7 + np.array([255, 0, 0]) * 0.3

# Visualization
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(color_mask)
plt.title("SAM Segmentation Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay.astype(np.uint8))
plt.title("Overlay (Original + Mask)")
plt.axis("off")

plt.tight_layout()
plt.show()