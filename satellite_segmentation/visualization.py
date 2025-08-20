"""Visualization utilities for segmentation results"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .processing import create_color_palette
from .constants import AVAILABLE_MODELS, CITYSCAPES_LABELS, get_ade

def create_visualization(original_image, mask, model_name, is_binary=False):
    """Create visualization plots"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Segmentation mask
    if is_binary:
        # Binary mask (SAM)
        color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        color_mask[mask == 1] = [255, 0, 0]  # Red for segmented area
        axes[1].imshow(color_mask)
    else:
        # Multi-class mask
        palette = create_color_palette(max(np.unique(mask)) + 1)
        color_mask = palette[mask]
        axes[1].imshow(color_mask)
    
    axes[1].set_title(f"Segmentation Mask\n{model_name}")
    axes[1].axis("off")
    
    # Overlay
    if is_binary:
        overlay_array = np.array(original_image)
        overlay = overlay_array.copy().astype(np.float32)
        overlay[mask == 1] = overlay[mask == 1] * 0.7 + np.array([255, 0, 0]) * 0.3
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    else:
        overlay_array = np.array(original_image)
        overlay = Image.blend(original_image, Image.fromarray(color_mask.astype(np.uint8)), alpha=0.6)
        overlay = np.array(overlay)
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    
    plt.tight_layout()
    return fig

def create_class_distribution_chart(mask, model_key=None):
    """Create class distribution bar chart with human-readable labels"""    
    unique_labels, counts = np.unique(mask, return_counts=True)
    max_class = int(max(unique_labels))

    # Auto-detect dataset if model_key not provided
    if model_key is None:
        if max_class <= 18:  # Likely Cityscapes (19 classes, 0-18)
            label_map = CITYSCAPES_LABELS
            dataset = "cityscapes"
        elif max_class <= 149:  # Likely ADE20K (150 classes, 0-149)
            label_map = {}  # Would need a model name to get ADE labels
            dataset = "ade20k"
        else:
            label_map = {}
            dataset = "unknown"
    else:
        # Use the previous logic with model_key
        model_info = None
        for category, models in AVAILABLE_MODELS.items():
            if model_key in models:
                model_info = models[model_key]
                break
        
        if model_info:
            dataset = model_info.get("dataset", "").lower()
            model_name = model_info.get("model_name")
            
            # Select label mapping
            if "cityscapes" in dataset:
                label_map = CITYSCAPES_LABELS
            elif "ade20k" in dataset:
                label_map = get_ade(model_name) or {}
            else:
                label_map = {}
        else:
            dataset = "unknown"
            label_map = {}

    # Convert IDs to labels (fallback to Class ID if missing)
    labels = [label_map.get(int(l), f"Class {l}") for l in unique_labels]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(labels)), counts)
    ax.set_xlabel("Class")
    ax.set_ylabel("Pixel Count")
    ax.set_title(f"Distribution of Classes in Segmentation ({dataset})")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{count}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig