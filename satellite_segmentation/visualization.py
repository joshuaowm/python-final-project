"""Visualization utilities for segmentation results"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .processing import create_color_palette

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

def create_class_distribution_chart(mask):
    """Create class distribution bar chart"""
    unique_labels, counts = np.unique(mask, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(unique_labels)), counts)
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Pixel Count")
    ax.set_title("Distribution of Classes in Segmentation")
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels(unique_labels)
    
    return fig