#!/usr/bin/env python3
"""
Satellite Image Semantic Segmentation Inference Script
Uses pretrained segmentation_models.pytorch for satellite RGB image analysis
Optimized for remote sensing applications including land cover classification
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms
import argparse


def load_model(architecture='Unet', encoder='resnet34', num_classes=21, checkpoint_path=None):
    """
    Load a pretrained segmentation model
    
    Args:
        architecture: Model architecture (Unet, FPN, Linknet, PSPNet, PAN, DeepLabV3, DeepLabV3Plus)
        encoder: Encoder backbone (resnet34, efficientnet-b4, etc.)
        num_classes: Number of segmentation classes
        checkpoint_path: Path to custom checkpoint (optional)
    
    Returns:
        model: Loaded segmentation model
    """
    # Create model
    if architecture.lower() == 'unet':
        model = smp.Unet(encoder_name=encoder, classes=num_classes, activation=None)
    elif architecture.lower() == 'fpn':
        model = smp.FPN(encoder_name=encoder, classes=num_classes, activation=None)
    elif architecture.lower() == 'linknet':
        model = smp.Linknet(encoder_name=encoder, classes=num_classes, activation=None)
    elif architecture.lower() == 'pspnet':
        model = smp.PSPNet(encoder_name=encoder, classes=num_classes, activation=None)
    elif architecture.lower() == 'pan':
        model = smp.PAN(encoder_name=encoder, classes=num_classes, activation=None)
    elif architecture.lower() == 'deeplabv3':
        model = smp.DeepLabV3(encoder_name=encoder, classes=num_classes, activation=None)
    elif architecture.lower() == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(encoder_name=encoder, classes=num_classes, activation=None)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    # Load custom checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
        print(f"Loaded custom checkpoint: {checkpoint_path}")
    else:
        print(f"Using pretrained {architecture} with {encoder} encoder")
    
    model.eval()
    return model


def preprocess_image(image_path, input_size=(512, 512), use_satellite_normalization=True):
    """
    Preprocess input image for inference (optimized for satellite imagery)
    
    Args:
        image_path: Path to input RGB image
        input_size: Target input size (height, width)
        use_satellite_normalization: Use satellite-specific normalization values
    
    Returns:
        image_tensor: Preprocessed image tensor
        original_image: Original PIL image
    """
    # Load image
    original_image = Image.open(image_path).convert('RGB')
    
    if use_satellite_normalization:
        # Satellite imagery normalization (typical values for Sentinel-2/Landsat)
        # These values work better for satellite RGB composites
        mean = [0.485, 0.456, 0.406]  # Can be adjusted based on your satellite data
        std = [0.229, 0.224, 0.225]   # Standard ImageNet values often work well
        
        # Alternative satellite-specific values (uncomment if needed):
        # mean = [0.5, 0.5, 0.5]  # More neutral for satellite data
        # std = [0.25, 0.25, 0.25]  # Adjusted for satellite imagery dynamic range
    else:
        # Standard ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    # Define preprocessing transforms
    preprocess = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Apply preprocessing
    image_tensor = preprocess(original_image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_image


def postprocess_output(output, original_size=None):
    """
    Postprocess model output to get segmentation mask
    
    Args:
        output: Raw model output tensor
        original_size: Original image size for resizing
    
    Returns:
        segmentation_mask: Segmentation mask as numpy array
    """
    # Apply softmax to get probabilities
    probabilities = torch.softmax(output, dim=1)
    
    # Get predicted class for each pixel
    segmentation_mask = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()
    
    # Resize to original size if specified
    if original_size:
        mask_pil = Image.fromarray(segmentation_mask.astype(np.uint8))
        mask_pil = mask_pil.resize(original_size, Image.NEAREST)
        segmentation_mask = np.array(mask_pil)
    
    return segmentation_mask


def visualize_results(original_image, segmentation_mask, class_names=None, save_path=None):
    """
    Visualize segmentation results
    
    Args:
        original_image: Original PIL image
        segmentation_mask: Segmentation mask
        class_names: List of class names (optional)
        save_path: Path to save visualization (optional)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(segmentation_mask, cmap='tab20')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay
    overlay = np.array(original_image)
    colored_mask = plt.cm.tab20(segmentation_mask / segmentation_mask.max())[:, :, :3]
    overlay_result = 0.6 * overlay/255.0 + 0.4 * colored_mask
    axes[2].imshow(overlay_result)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def run_inference(image_path, model_config=None, save_results=False):
    """
    Run inference on a single RGB image
    
    Args:
        image_path: Path to input image
        model_config: Dictionary with model configuration
        save_results: Whether to save results
    """
    # Default model configuration
    if model_config is None:
        model_config = {
            'architecture': 'Unet',
            'encoder': 'resnet34',
            'num_classes': 21,  # PASCAL VOC classes
            'input_size': (512, 512)
        }
    
    print(f"Loading model: {model_config['architecture']} with {model_config['encoder']}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(
        architecture=model_config['architecture'],
        encoder=model_config['encoder'],
        num_classes=model_config['num_classes'],
        checkpoint_path=model_config.get('checkpoint_path')
    )
    model = model.to(device)
    
    # Preprocess image
    print(f"Preprocessing image: {image_path}")
    image_tensor, original_image = preprocess_image(
        image_path, 
        input_size=model_config['input_size']
    )
    image_tensor = image_tensor.to(device)
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = model(image_tensor)
    
    # Postprocess output
    segmentation_mask = postprocess_output(output, original_image.size)
    
    # PASCAL VOC class names (if using 21 classes)
    pascal_voc_classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
        'sofa', 'train', 'tvmonitor'
    ]
    
    class_names = pascal_voc_classes if model_config['num_classes'] == 21 else None
    
    # Visualize results
    save_path = None
    if save_results:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = f"{base_name}_segmentation_result.png"
    
    visualize_results(original_image, segmentation_mask, class_names, save_path)
    
    # Print some statistics
    unique_classes = np.unique(segmentation_mask)
    print(f"\nDetected classes: {unique_classes}")
    if class_names:
        detected_class_names = [class_names[i] for i in unique_classes if i < len(class_names)]
        print(f"Class names: {detected_class_names}")
    
    return segmentation_mask, original_image


def main():
    parser = argparse.ArgumentParser(description='Satellite Image Semantic Segmentation Inference')
    parser.add_argument('--image', required=True, help='Path to input satellite RGB image')
    parser.add_argument('--architecture', default='Unet', 
                       choices=['Unet', 'FPN', 'Linknet', 'PSPNet', 'PAN', 'DeepLabV3', 'DeepLabV3Plus'],
                       help='Model architecture')
    parser.add_argument('--encoder', default='resnet34', 
                       help='Encoder backbone (resnet34, resnet50, efficientnet-b4, etc.)')
    parser.add_argument('--num_classes', type=int, default=6, 
                       help='Number of classes (6 for basic satellite classes, 10 for extended)')
    parser.add_argument('--input_size', nargs=2, type=int, default=[512, 512], 
                       help='Input size (height width). Try 768 768 for better satellite results')
    parser.add_argument('--checkpoint', help='Path to custom trained checkpoint')
    parser.add_argument('--save', action='store_true', help='Save visualization results')
    parser.add_argument('--no_satellite_norm', action='store_true', 
                       help='Disable satellite-specific normalization')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Satellite image file not found: {args.image}")
        return
    
    # Model configuration for satellite imagery
    model_config = {
        'architecture': args.architecture,
        'encoder': args.encoder,
        'num_classes': args.num_classes,
        'input_size': tuple(args.input_size),
        'checkpoint_path': args.checkpoint,
        'use_satellite_normalization': not args.no_satellite_norm
    }
    
    print("=== Satellite Image Segmentation ===")
    print(f"Image: {args.image}")
    print(f"Model: {args.architecture} + {args.encoder}")
    print(f"Classes: {args.num_classes}")
    print(f"Input size: {args.input_size}")
    
    # Run inference
    try:
        mask, image = run_inference(args.image, model_config, args.save)
        print("\nSatellite image segmentation completed successfully!")
    except Exception as e:
        print(f"Error during inference: {str(e)}")


if __name__ == "__main__":
    # Example usage when run directly
    if len(os.sys.argv) == 1:
        print("=== Satellite Image Segmentation Examples ===")
        print("Basic usage:")
        print("python segmentation_inference.py --image satellite_image.jpg")
        print("\nAdvanced usage:")
        print("python segmentation_inference.py --image satellite.tif --architecture DeepLabV3Plus --encoder efficientnet-b4 --input_size 768 768 --num_classes 10")
        print("\nWith custom trained model:")
        print("python segmentation_inference.py --image satellite.jpg --checkpoint my_satellite_model.pth --num_classes 8")
    else:
        main()