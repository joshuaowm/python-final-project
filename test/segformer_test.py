import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# More checkpoints can be found here:
# https://huggingface.co/collections/smp-hub/segformer-6749eb4923dea2c355f29a1f
checkpoint = "smp-hub/segformer-b2-1024x1024-city-160k"

# Load pretrained model and preprocessing function
model = smp.from_pretrained(checkpoint).eval().to(device)
preprocessing = A.Compose.from_pretrained(checkpoint)

# Load image from local pathw
image_path = "./test/test2.png"  # Replace with your image path

# Check if image exists
if not os.path.exists(image_path):
    print(f"Error: Image file not found: {image_path}")
    exit(1)

# Load image
image = Image.open(image_path)
print(f"Loaded image: {image_path}")
print(f"Image size: {image.size}")

# Convert to RGB if needed (in case of RGBA or other formats)
if image.mode != 'RGB':
    image = image.convert('RGB')

# Preprocess image
image = np.array(image)
normalized_image = preprocessing(image=image)["image"]
input_tensor = torch.as_tensor(normalized_image)
input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
input_tensor = input_tensor.to(device)

print(f"Input tensor shape: {input_tensor.shape}")

# Perform inference
with torch.inference_mode():
    output_mask = model(input_tensor)

print(f"Output mask shape: {output_mask.shape}")

# Postprocess mask
mask = torch.nn.functional.interpolate(
    output_mask, size=image.shape[:2], mode="bilinear", align_corners=False
)
mask = mask[0].argmax(0).cpu().numpy()

print(f"Final mask shape: {mask.shape}")
print(f"Unique classes in mask: {np.unique(mask)}")

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.axis("off")
plt.imshow(image)
plt.title(f"Input Image\n{os.path.basename(image_path)}")

plt.subplot(122)
plt.axis("off")
plt.imshow(mask, cmap='tab20')
plt.title("Output Mask")

plt.tight_layout()
plt.show()