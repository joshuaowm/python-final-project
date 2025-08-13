import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BeitFeatureExtractor, BeitForSemanticSegmentation
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model & feature extractor
checkpoint = "microsoft/beit-base-finetuned-ade-640-640"
feature_extractor = BeitFeatureExtractor.from_pretrained(checkpoint)
model = BeitForSemanticSegmentation.from_pretrained(checkpoint).to(device).eval()

# Load image
image_path = "./test/test.png"  # Change to your image path
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found: {image_path}")

image = Image.open(image_path)
if image.mode != "RGB":
    image = image.convert("RGB")

print(f"Loaded image: {image_path}")
print(f"Image size: {image.size}")

# Preprocess image
inputs = feature_extractor(images=image, return_tensors="pt").to(device)

# Run inference & time it
start_time = time.time()

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits  # Shape: [batch, num_classes, height, width]

inference_time = time.time() - start_time
print(f"Inference completed in {inference_time:.2f} seconds")

# Postprocess mask
# -----------------------------
# Resize logits to original image size
logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],  # (H, W)
    mode="bilinear",
    align_corners=False
)
predicted_mask = logits.argmax(dim=1)[0].cpu().numpy()

print(f"Mask shape: {predicted_mask.shape}")
print(f"Unique classes in mask: {np.unique(predicted_mask)}")

# ADE20K color palette (extended version)
def ade_palette():
    """ADE20K 150-class palette."""
    return [
        [120, 120, 120], [180, 120, 120], [  6, 230, 230], [ 80,  50,  50],
        [  4, 200,   3], [120, 120,  80], [140, 140, 140], [204,   5, 255],
        [230, 230, 230], [  4, 250,   7], [224,   5, 255], [235, 255,   7],
        [150,   5,  61], [120, 120,  70], [  8, 255,  51], [255,   6,  82],
        [143, 255, 140], [ 204,  255,   4], [255,  51,   7], [204,  70,   3],
        [  0, 102, 200], [ 61, 230, 250], [255,   6, 51], [11, 102, 255],
        [255, 7,  71], [255,   9, 224], [  9,   7, 230], [220, 220, 220],
        [255,   9,  92], [112,   9, 255], [  8, 255, 214], [  7, 255, 224],
        [255, 184,   6], [  10, 255,  71], [255,  41,  10], [  7, 255,  255],
        [224, 255,   8], [102,   8, 255], [255,  61,   6], [255, 194,   7],
        [255, 122,   8], [  0, 255,  20], [255,   8,  41], [255,   5, 153],
        [  6,  51, 255], [235,  12, 255], [160, 150,  20], [  0, 163, 255],
        [140,  140, 140], [250,  10,  15], [  20, 255,   0], [  31, 255,   0],
        [255,  31,   0], [255, 224,   0], [153, 255,   0], [  0,   0, 255],
        [255,  71,   0], [  0, 235, 255], [  0, 173, 255], [  31,   0, 255],
        [  11,  200, 200], [255, 82,   0], [  0, 255, 245], [  0,  61, 255],
        [  0, 255, 112], [  0, 255, 133], [255,   0,   0], [255, 163,   0],
        [255, 102,   0], [194, 255,   0], [  0, 143, 255], [  51, 255,   0],
        [  0,  82, 255], [  0, 255,  41], [ 0,  255, 173], [  10,   0, 255],
        [173, 255,   0], [  0, 255,  153], [255,  92,   0], [255,   0,  255],
        # Add more colors to reach 150 classes
    ]

# Generate a full palette with 150+ colors
def generate_full_palette(num_classes=150):
    """Generate a colorful palette for semantic segmentation."""
    np.random.seed(42)  # For reproducible colors
    palette = []
    
    # Add the predefined ADE20K colors
    base_colors = ade_palette()
    palette.extend(base_colors)
    
    # Generate additional random colors if needed
    while len(palette) < num_classes:
        color = np.random.randint(0, 256, 3).tolist()
        palette.append(color)
    
    return palette[:num_classes]

# Create color mask
palette = generate_full_palette(max(np.unique(predicted_mask)) + 1)
palette = np.array(palette)

color_mask = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1], 3), dtype=np.uint8)

for label in np.unique(predicted_mask):
    if label < len(palette):
        color_mask[predicted_mask == label] = palette[label]

# Create overlay for better visualization
image_array = np.array(image)
overlay = image_array.copy().astype(np.float32)

# Create a blended overlay (70% original image + 30% color mask)
alpha = 0.6  # transparency of the segmentation mask
overlay = (1 - alpha) * image_array + alpha * color_mask
overlay = np.clip(overlay, 0, 255)  # Ensure values are in valid range

# Visualization (side-by-side)
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(color_mask)
plt.title("BEiT Segmentation Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(overlay.astype(np.uint8))
plt.title("Overlay (Original + Mask)")
plt.axis("off")

plt.tight_layout()
plt.show()

# Optional: Print some statistics
print(f"Number of unique classes detected: {len(np.unique(predicted_mask))}")
print(f"Class distribution: {np.bincount(predicted_mask.flatten())[:10]}")  # Show first 10 classes
