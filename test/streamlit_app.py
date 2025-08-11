#!/usr/bin/env python3
"""
Streamlit App for Satellite Image Semantic Segmentation
Interactive web interface for satellite RGB image analysis using segmentation_models_pytorch
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision import transforms
import io
import os
from typing import Optional, Tuple, Dict


# Page configuration
st.set_page_config(
    page_title="Satellite Image Segmentation",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading to avoid reloading on every interaction
@st.cache_resource
def load_segmentation_model(architecture: str, encoder: str, num_classes: int, activation: Optional[str] = None):
    """
    Load a pretrained segmentation model with caching
    
    Args:
        architecture: Model architecture
        encoder: Encoder backbone
        num_classes: Number of segmentation classes
        activation: Activation function (optional)
    
    Returns:
        model: Loaded segmentation model
    """
    try:
        # Create model based on architecture
        if architecture.lower() == 'unet':
            model = smp.Unet(encoder_name=encoder, classes=num_classes, activation=activation)
        elif architecture.lower() == 'fpn':
            model = smp.FPN(encoder_name=encoder, classes=num_classes, activation=activation)
        elif architecture.lower() == 'linknet':
            model = smp.Linknet(encoder_name=encoder, classes=num_classes, activation=activation)
        elif architecture.lower() == 'pspnet':
            model = smp.PSPNet(encoder_name=encoder, classes=num_classes, activation=activation)
        elif architecture.lower() == 'pan':
            model = smp.PAN(encoder_name=encoder, classes=num_classes, activation=activation)
        elif architecture.lower() == 'deeplabv3':
            model = smp.DeepLabV3(encoder_name=encoder, classes=num_classes, activation=activation)
        elif architecture.lower() == 'deeplabv3plus':
            model = smp.DeepLabV3Plus(encoder_name=encoder, classes=num_classes, activation=activation)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


def preprocess_image(image: Image.Image, input_size: Tuple[int, int] = (512, 512), 
                    use_satellite_normalization: bool = True) -> torch.Tensor:
    """
    Preprocess input image for inference
    
    Args:
        image: PIL Image
        input_size: Target input size (height, width)
        use_satellite_normalization: Use satellite-specific normalization
    
    Returns:
        image_tensor: Preprocessed image tensor
    """
    if use_satellite_normalization:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    
    preprocess = transforms.Compose([
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor


def postprocess_output(output: torch.Tensor, original_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Postprocess model output to get segmentation mask
    
    Args:
        output: Raw model output tensor
        original_size: Original image size for resizing
    
    Returns:
        segmentation_mask: Segmentation mask as numpy array
    """
    probabilities = torch.softmax(output, dim=1)
    segmentation_mask = torch.argmax(probabilities, dim=1).squeeze(0).cpu().numpy()
    
    if original_size:
        mask_pil = Image.fromarray(segmentation_mask.astype(np.uint8))
        mask_pil = mask_pil.resize(original_size, Image.NEAREST)
        segmentation_mask = np.array(mask_pil)
    
    return segmentation_mask


def create_visualization(original_image: Image.Image, segmentation_mask: np.ndarray, 
                        class_names: list = None) -> plt.Figure:
    """
    Create visualization of segmentation results
    
    Args:
        original_image: Original PIL image
        segmentation_mask: Segmentation mask
        class_names: List of class names
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Segmentation mask
    im = axes[1].imshow(segmentation_mask, cmap='tab20')
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    # Overlay
    overlay = np.array(original_image)
    colored_mask = plt.cm.tab20(segmentation_mask / max(segmentation_mask.max(), 1))[:, :, :3]
    overlay_result = 0.6 * overlay/255.0 + 0.4 * colored_mask
    axes[2].imshow(overlay_result)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def get_class_names(num_classes: int) -> list:
    """Get class names based on number of classes"""
    if num_classes == 2:
        return ['background', 'foreground']
    elif num_classes == 21:
        return [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
            'sofa', 'train', 'tvmonitor'
        ]
    elif num_classes == 6:
        return ['background', 'vegetation', 'water', 'urban', 'bare_soil', 'cloud']
    elif num_classes == 10:
        return ['background', 'forest', 'grassland', 'water', 'urban', 'agricultural', 
                'bare_soil', 'wetland', 'cloud', 'snow']
    else:
        return [f'class_{i}' for i in range(num_classes)]


def main():
    st.title("🛰️ Satellite Image Semantic Segmentation")
    st.markdown("Upload a satellite RGB image and configure the segmentation model parameters.")
    
    # Sidebar for model configuration
    st.sidebar.header("🔧 Model Configuration")
    
    # Architecture selection
    architecture = st.sidebar.selectbox(
        "Architecture",
        ['Unet', 'FPN', 'Linknet', 'PSPNet', 'PAN', 'DeepLabV3', 'DeepLabV3Plus'],
        index=0,
        help="Choose the segmentation model architecture"
    )
    
    # Encoder selection
    encoder = st.sidebar.selectbox(
        "Encoder",
        ['resnet34', 'resnet50', 'resnet101', 'efficientnet-b0', 'efficientnet-b4', 
         'efficientnet-b7', 'resnext50_32x4d', 'timm-regnety_002', 'timm-efficientnet-b0'],
        index=0,
        help="Choose the encoder backbone"
    )
    
    # Number of classes
    num_classes = st.sidebar.number_input(
        "Number of Classes",
        min_value=2,
        max_value=100,
        value=6,
        step=1,
        help="Number of segmentation classes (2-100). Common values: 2=binary, 6=basic land cover, 21=PASCAL VOC"
    )
    
    # Activation function
    activation_options = [None, 'sigmoid', 'softmax', 'logsoftmax', 'tanh', 'identity']
    activation = st.sidebar.selectbox(
        "Activation Function (Optional)",
        activation_options,
        index=0,
        help="Activation function for model output"
    )
    
    # Input size configuration
    st.sidebar.subheader("Input Size")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        input_height = st.number_input("Height", min_value=256, max_value=1024, value=512, step=32)
    with col2:
        input_width = st.number_input("Width", min_value=256, max_value=1024, value=512, step=32)
    
    # Advanced options
    st.sidebar.subheader("Advanced Options")
    use_satellite_norm = st.sidebar.checkbox("Use Satellite Normalization", value=True,
                                           help="Use satellite-specific normalization values")
    resize_to_original = st.sidebar.checkbox("Resize Output to Original Size", value=True)
    
    # Main content area
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose a satellite image...",
            type=['jpg', 'jpeg', 'png', 'tiff', 'tif'],
            help="Upload RGB satellite imagery (JPG, PNG, or TIFF format)"
        )
    with col2:
        if uploaded_file is not None:
            # Display image information
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption=f"Uploaded Image", use_container_width=True)
            
            # Image information
            st.write(f"**Image Size:** {image.size[0]} × {image.size[1]} pixels")
            st.write(f"**File Size:** {len(uploaded_file.getvalue()) / (1024*1024):.2f} MB")
            st.write(f"**Format:** {image.format if hasattr(image, 'format') else 'Unknown'}")


    st.subheader("🎯 Segmentation Results")
    
    if uploaded_file is not None:
        if st.button("🚀 Run Segmentation", type="primary"):
            try:
                with st.spinner("Loading model and processing image..."):
                    # Set device
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    st.info(f"Using device: {device}")
                    
                    # Load model
                    model = load_segmentation_model(architecture, encoder, num_classes, activation)
                    if model is None:
                        st.error("Failed to load model. Please check your configuration.")
                        return
                    
                    model = model.to(device)
                    
                    # Preprocess image
                    input_size = (input_height, input_width)
                    image_tensor = preprocess_image(image, input_size, use_satellite_norm)
                    image_tensor = image_tensor.to(device)
                    
                    # Run inference
                    with torch.no_grad():
                        output = model(image_tensor)
                    
                    # Postprocess output
                    original_size = image.size if resize_to_original else None
                    segmentation_mask = postprocess_output(output, original_size)
                    
                    # Get class names
                    class_names = get_class_names(num_classes)
                    
                    # Create and display visualization
                    fig = create_visualization(image, segmentation_mask, class_names)
                    st.pyplot(fig)
                    
                    # Display statistics
                    st.subheader("📊 Segmentation Statistics")
                    unique_classes = np.unique(segmentation_mask)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Detected Classes:** {len(unique_classes)}")
                        st.write(f"**Class IDs:** {list(unique_classes)}")
                    
                    with col2:
                        st.write("**Class Distribution:**")
                        for class_id in unique_classes:
                            pixel_count = np.sum(segmentation_mask == class_id)
                            percentage = (pixel_count / segmentation_mask.size) * 100
                            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                            st.write(f"- {class_name}: {percentage:.1f}%")
                    
                    # Download option for segmentation mask
                    st.subheader("💾 Download Results")
                    
                    # Convert mask to PIL Image for download
                    mask_image = Image.fromarray((segmentation_mask * 255 / segmentation_mask.max()).astype(np.uint8))
                    
                    # Create download buffer
                    buf = io.BytesIO()
                    mask_image.save(buf, format='PNG')
                    buf.seek(0)
                    
                    st.download_button(
                        label="Download Segmentation Mask",
                        data=buf,
                        file_name="segmentation_mask.png",
                        mime="image/png"
                    )
                    
            except Exception as e:
                st.error(f"Error during segmentation: {str(e)}")
                st.write("Please check your model configuration and try again.")
    else:
        st.info("👆 Upload an image to begin segmentation")
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About This Tool")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Model Architectures:**
        - UNet: Classic encoder-decoder
        - FPN: Feature Pyramid Network
        - LinkNet: Efficient architecture
        - PSPNet: Pyramid Scene Parsing
        - PAN: Pyramid Attention Network
        - DeepLabV3/V3+: Atrous convolution
        """)
    
    with col2:
        st.markdown("""
        **Common Use Cases:**
        - Land cover classification
        - Urban planning analysis
        - Environmental monitoring
        - Agricultural assessment
        - Water body detection
        - Cloud/shadow masking
        """)
    
    with col3:
        st.markdown("""
        **Tips for Best Results:**
        - Use 768×768 input size for detailed results
        - Try EfficientNet encoders for accuracy
        - Use satellite normalization for aerial/satellite imagery
        - Adjust number of classes based on your dataset
        - Consider DeepLabV3+ for complex scenes
        """)


if __name__ == "__main__":
    main()