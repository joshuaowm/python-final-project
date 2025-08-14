"""
Semantic Segmentation Web App using Streamlit
Combines multiple models: SegFormer, BEiT, DPT, SAM, and SMP
"""

import streamlit as st
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import tempfile

# Transformers imports
from transformers import (
    AutoImageProcessor, 
    AutoModelForSemanticSegmentation,
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    BeitImageProcessor,
    BeitForSemanticSegmentation,
    BeitFeatureExtractor,
    DPTImageProcessor,
    DPTForSemanticSegmentation,
    SamProcessor, 
    SamModel
)

# Additional imports
import albumentations as A
try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    st.warning("segmentation_models_pytorch not available. SMP models will be disabled.")

# Page config
st.set_page_config(
    page_title="Semantic Segmentation Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Available models configuration
AVAILABLE_MODELS = {
    "SegFormer Models": {
        "segformer-b5-cityscapes": {
            "model_name": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
            "description": "SegFormer-B5 Cityscapes (19 classes)",
            "type": "transformers"
        },
        "segformer-b2-ade": {
            "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",
            "description": "SegFormer-B2 ADE20K (150 classes)",
            "type": "transformers"
        },
    },
    "BEiT Models": {
        "beit-base-ade": {
            "model_name": "microsoft/beit-base-finetuned-ade-640-640",
            "description": "BEiT-Base ADE20K (150 classes)",
            "type": "transformers"
        }
    },
    "DPT Models": {
        "dpt-large-ade": {
            "model_name": "Intel/dpt-large-ade",
            "description": "DPT-Large ADE20K (150 classes)",
            "type": "transformers"
        }
    },
    "SAM Models": {
        "sam-vit-large": {
            "model_name": "facebook/sam-vit-large",
            "description": "SAM ViT-Large (Segment Anything)",
            "type": "sam"
        }
    }
}

# Add SMP models if available
if SMP_AVAILABLE:
    AVAILABLE_MODELS["SMP Models"] = {
        "smp-segformer-city": {
            "model_name": "smp-hub/segformer-b2-1024x1024-city-160k",
            "description": "SMP SegFormer-B2 Cityscapes",
            "type": "smp"
        }
    }

@st.cache_resource
def load_transformers_model(model_name, model_type):
    """Load Transformers model with caching"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        if "dpt" in model_name.lower():
            processor = DPTImageProcessor.from_pretrained(model_name)
            model = DPTForSemanticSegmentation.from_pretrained(model_name)
        elif "segformer" in model_name.lower():
            processor = SegformerImageProcessor.from_pretrained(model_name)
            model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        elif "beit" in model_name.lower():
            # Try both BeitImageProcessor and BeitFeatureExtractor
            try:
                processor = BeitImageProcessor.from_pretrained(model_name)
            except:
                processor = BeitFeatureExtractor.from_pretrained(model_name)
            model = BeitForSemanticSegmentation.from_pretrained(model_name)
        else:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        
        model = model.to(device).eval()
        return model, processor, device
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_resource
def load_sam_model(model_name):
    """Load SAM model with caching"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        processor = SamProcessor.from_pretrained(model_name)
        model = SamModel.from_pretrained(model_name).to(device).eval()
        return model, processor, device
    except Exception as e:
        st.error(f"Error loading SAM model: {e}")
        return None, None, None

@st.cache_resource
def load_smp_model(model_name):
    """Load SMP model with caching"""
    if not SMP_AVAILABLE:
        st.error("segmentation_models_pytorch not available")
        return None, None, None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = smp.from_pretrained(model_name).eval().to(device)
        preprocessing = A.Compose.from_pretrained(model_name)
        return model, preprocessing, device
    except Exception as e:
        st.error(f"Error loading SMP model: {e}")
        return None, None, None

def create_color_palette(num_classes=150):
    """Create color palette for visualization"""
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # Background as black
    return palette

def process_transformers_model(image, model, processor, device):
    """Process image with Transformers models"""
    original_size = image.size
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - start_time
    
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
    
    return predicted_mask, inference_time

def process_sam_model(image, model, processor, device):
    """Process image with SAM model"""
    # SAM needs prompts - use center point
    input_points = [[[image.width // 2, image.height // 2]]]
    
    inputs = processor(
        images=image,
        input_points=input_points,
        return_tensors="pt"
    ).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    inference_time = time.time() - start_time
    
    masks = outputs.pred_masks
    mask = masks[0, 0, 0].cpu().numpy()
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Resize if needed
    if binary_mask.shape != (image.height, image.width):
        from scipy.ndimage import zoom
        zoom_h = image.height / binary_mask.shape[0]
        zoom_w = image.width / binary_mask.shape[1]
        binary_mask = zoom(binary_mask, (zoom_h, zoom_w), order=0).astype(np.uint8)
    
    return binary_mask, inference_time

def process_smp_model(image, model, preprocessing, device):
    """Process image with SMP model"""
    # Convert to numpy array
    image_array = np.array(image)
    
    # Preprocess
    normalized_image = preprocessing(image=image_array)["image"]
    input_tensor = torch.as_tensor(normalized_image)
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    # Inference
    start_time = time.time()
    with torch.inference_mode():
        output_mask = model(input_tensor)
    inference_time = time.time() - start_time
    
    # Post-process
    mask = torch.nn.functional.interpolate(
        output_mask, size=image_array.shape[:2], mode="bilinear", align_corners=False
    )
    mask = mask[0].argmax(0).cpu().numpy()
    
    return mask, inference_time

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
        num_classes = len(np.unique(mask))
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

def main():
    st.title("🎯 Semantic Segmentation Hub")
    st.markdown("Upload an image and choose a model to perform semantic segmentation")
    
    # Sidebar
    st.sidebar.header("Model Selection")
    
    # Flatten model options for selectbox
    model_options = []
    model_mapping = {}
    
    for category, models in AVAILABLE_MODELS.items():
        for key, config in models.items():
            display_name = f"{category}: {config['description']}"
            model_options.append(display_name)
            model_mapping[display_name] = (key, config)
    
    selected_display = st.sidebar.selectbox("Choose a model:", model_options)
    selected_key, selected_config = model_mapping[selected_display]
    
    # Device info
    device_name = "GPU" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"Running on: {device_name}")
    
    # Model info
    st.sidebar.markdown("### Model Info")
    st.sidebar.write(f"**Model:** {selected_config['description']}")
    st.sidebar.write(f"**Type:** {selected_config['type'].upper()}")
    
    col1, col2 = st.columns([2,1])
    with col1:
        # File upload
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])

    with col2:
        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.write(f"Image size: {image.size}")
    
    with col1:
        if uploaded_file is not None:
            if st.button("Run Segmentation", type="primary"):
                with st.spinner("Loading model..."):
                    # Load model based on type
                    model_name = selected_config["model_name"]
                    model_type = selected_config["type"]
                    
                    if model_type == "transformers":
                        model, processor, device = load_transformers_model(model_name, model_type)
                    elif model_type == "sam":
                        model, processor, device = load_sam_model(model_name)
                    elif model_type == "smp":
                        model, processor, device = load_smp_model(model_name)
                    else:
                        st.error(f"Unknown model type: {model_type}")
                        return
                    
                    if model is None:
                        st.error("Failed to load model")
                        return
                
                with st.spinner("Running inference..."):
                    # Process image
                    if model_type == "transformers":
                        mask, inference_time = process_transformers_model(image, model, processor, device)
                        is_binary = False
                    elif model_type == "sam":
                        mask, inference_time = process_sam_model(image, model, processor, device)
                        is_binary = True
                    elif model_type == "smp":
                        mask, inference_time = process_smp_model(image, model, processor, device)
                        is_binary = False
                
                # Display results
                st.success(f"Segmentation completed in {inference_time:.2f} seconds")
                
                # Create and display visualization
                fig = create_visualization(image, mask, selected_config['description'], is_binary)
                st.pyplot(fig)
                
                # Statistics
                st.header("Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Inference Time", f"{inference_time:.2f}s")
                
                with col2:
                    unique_classes = len(np.unique(mask))
                    st.metric("Unique Classes", unique_classes)
                
                with col3:
                    if is_binary:
                        segmented_pixels = np.sum(mask == 1)
                        total_pixels = mask.size
                        percentage = (segmented_pixels / total_pixels) * 100
                        st.metric("Segmented Area", f"{percentage:.1f}%")
                    else:
                        st.metric("Mask Shape", f"{mask.shape[0]}×{mask.shape[1]}")
                
                # Class distribution (for multi-class models)
                if not is_binary and unique_classes > 1:
                    st.header("Class Distribution")
                    unique_labels, counts = np.unique(mask, return_counts=True)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(range(len(unique_labels)), counts)
                    ax.set_xlabel("Class ID")
                    ax.set_ylabel("Pixel Count")
                    ax.set_title("Distribution of Classes in Segmentation")
                    ax.set_xticks(range(len(unique_labels)))
                    ax.set_xticklabels(unique_labels)
                    
                    st.pyplot(fig)
    
    
    # Instructions
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        st.markdown("""
        ### Instructions:
        1. **Select a model** from the sidebar
        2. **Upload an image** using the file uploader
        3. **Click "Run Segmentation"** to process the image
        4. **View results** including the original image, segmentation mask, and overlay
        """)
    with c2:
        st.markdown("""
        ### Available Models:
        - **SegFormer**: Transformer-based models for semantic segmentation
        - **BEiT**: BERT pre-trained image transformer
        - **DPT**: Dense Prediction Transformer
        - **SAM**: Segment Anything Model (requires prompts)
        - **SMP**: Segmentation Models PyTorch library models
        """)
    with c3:
        st.markdown("""
        ### Tips:
        - Different models are trained on different datasets (Cityscapes, ADE20K)
        - SAM works differently - it segments objects based on prompts (uses center point)
        - GPU acceleration will be used if available
        """)

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and PyTorch")

if __name__ == "__main__":
    main()