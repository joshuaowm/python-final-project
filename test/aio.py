""" 
Semantic segmentation using Transformers models. 
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    AutoImageProcessor, 
    AutoModelForSemanticSegmentation,
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    BeitImageProcessor,
    BeitForSemanticSegmentation,
    DPTImageProcessor,
    DPTForSemanticSegmentation
)
import sys
import os
import argparse

# Available models
AVAILABLE_MODELS = {
    "segformer-b5-cityscapes": {
        "model_name": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        "description": "SegFormer-B5 Cityscapes",
        "classes": "Cityscapes (19 classes)",
    },
    "segformer-b2-ade": {
        "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",
        "description": "SegFormer-B2 ADE20K",
        "classes": "ADE20K (150 classes)",
    },
    "beit-base": {
        "model_name": "microsoft/beit-base-finetuned-ade-640-640",
        "description": "BEiT-Base",
        "classes": "ADE20K (150 classes)",
    },
    "upernet-swin-base": {
        "model_name": "openmmlab/upernet-swin-base",
        "description": "UperNet-Swin-base",
        "classes": "ADE20K (150 classes)",
    },
    "dpt-large": {
        "model_name": "Intel/dpt-large-ade",
        "description": "DPT-Large",
        "classes": "ADE20K (150 classes)",
    }
}

def load_model(model_key, device):
    """Load model and processor"""
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}")
    
    model_name = AVAILABLE_MODELS[model_key]["model_name"]
    print(f"Loading {model_key}: {model_name}")
        
    try:
        if "dpt" in model_name.lower():
            processor = DPTImageProcessor.from_pretrained(model_name)
            model = DPTForSemanticSegmentation.from_pretrained(model_name)
        elif "segformer" in model_name.lower():
            processor = SegformerImageProcessor.from_pretrained(model_name)
            model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        elif "beit" in model_name.lower():
            processor = BeitImageProcessor.from_pretrained(model_name)
            model = BeitForSemanticSegmentation.from_pretrained(model_name)
        else:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        
        model = model.to(device).eval()
        print(f"Model loaded on {device}")
        return model, processor, model_name
        
    except Exception as e:
        print(f"Error loading model: {e}")
        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
            model = model.to(device).eval()
            print(f"Model loaded with AutoClasses on {device}")
            return model, processor, model_name
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            raise

def create_color_palette(num_classes=150):
    """Create color palette for visualization"""
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # Background as black
    return palette

def process_image(image_path, model_key="segformer-b2-ade", device=None, save_results=False):
    """Process image with specified model"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    start_time = time.time()
    model, processor, model_name = load_model(model_key, device)
    load_time = time.time() - start_time
    
    # Load image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path}, size: {image.size}")
    original_size = image.size
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    print("Running inference...")
    inference_start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - inference_start
    
    # Post-process
    logits = outputs.logits
    
    # Resize to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=original_size[::-1],
        mode="bilinear",
        align_corners=False
    )
    
    predicted_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    # Create color visualization
    num_classes = logits.shape[1]
    palette = create_color_palette(num_classes)
    color_mask = palette[predicted_mask]
    
    # Create overlay
    overlay = Image.blend(image, Image.fromarray(color_mask.astype(np.uint8)), alpha=0.6)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(color_mask)
    axes[1].set_title(f"Segmentation Mask\n{AVAILABLE_MODELS[model_key]['description']}")
    axes[1].axis("off")
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()
    
    # Analysis
    unique_classes, counts = np.unique(predicted_mask, return_counts=True)
    print(f"\nInference time: {inference_time:.2f} seconds")
    print(f"Model loading time: {load_time:.2f} seconds")
    print(f"Total process time: {(time.time() - start_time):.2f} seconds")
    print(f"Found {len(unique_classes)} different classes")
    
    # Save results
    if save_results:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        model_short = model_key.replace("-", "_")
        
        mask_path = f"{base_name}_{model_short}_mask.png"
        Image.fromarray(color_mask.astype(np.uint8)).save(mask_path)
        print(f"Mask saved: {mask_path}")
        
        overlay_path = f"{base_name}_{model_short}_overlay.png"
        overlay.save(overlay_path)
        print(f"Overlay saved: {overlay_path}")
    
    return predicted_mask, color_mask, overlay

def main():
    parser = argparse.ArgumentParser(description="Semantic segmentation using Transformers models")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--model", "-m", default="segformer-b2-ade", 
                       choices=list(AVAILABLE_MODELS.keys()),
                       help="Model to use for segmentation")
    parser.add_argument("--save", "-s", action="store_true",
                       help="Save segmentation results")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    try:
        print(f"Starting segmentation with {args.model}")
        print(f"Image: {args.image_path}")
        print(f"Device: {device}")
        print("-" * 50)
        
        process_image(
            image_path=args.image_path,
            model_key=args.model,
            device=device,
            save_results=args.save
        )
        
        print("\nSegmentation completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    main()