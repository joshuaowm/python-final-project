import streamlit as st
import torch
import torchvision.transforms as T
from torchgeo.models import ResNet18_Weights, resnet18
from torchgeo.datasets import unbind_samples
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import os
from tempfile import NamedTemporaryFile

# Set page config
st.set_page_config(
    page_title="TorchGeo Image Analysis",
    page_icon="🛰️",
    layout="wide"
)

# App title and description
st.title("🛰️ TorchGeo Image Analysis")
st.markdown("""
Upload an image and apply a pre-trained TorchGeo model to analyze it. 
You can customize various settings to see how they affect the results.
""")

# Sidebar for model settings
st.sidebar.header("Model Settings")

model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["resnet18"]
)

pretrained = st.sidebar.checkbox("Use Pre-trained Weights", value=True)

# Add more advanced settings if model is selected
if model_type == "resnet18":
    num_classes = st.sidebar.slider("Number of Output Classes", min_value=1, max_value=100, value=21)
    in_channels = st.sidebar.selectbox("Input Channels", [3, 1, 4], index=0)

# Image preprocessing settings
st.sidebar.header("Image Preprocessing")
normalize = st.sidebar.checkbox("Normalize Image", value=True)
resize_method = st.sidebar.radio("Resize Method", ["Scale", "Crop"])
size = st.sidebar.slider("Image Size", min_value=64, max_value=1024, value=224, step=32)
apply_augmentation = st.sidebar.checkbox("Apply Data Augmentation", value=False)

# Function to load and preprocess image
def preprocess_image(uploaded_image, settings):
    img = Image.open(uploaded_image)
    
    # Create transformation pipeline
    transforms_list = []
    
    # Resize
    if settings["resize_method"] == "Scale":
        transforms_list.append(T.Resize((settings["size"], settings["size"])))
    else:
        transforms_list.append(T.CenterCrop((settings["size"], settings["size"])))
    
    # Convert to tensor
    transforms_list.append(T.ToTensor())
    
    # Apply normalization if selected
    if settings["normalize"]:
        transforms_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    
    # Data augmentation if selected
    if settings["apply_augmentation"]:
        augmentation = [
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ]
        transforms_list = augmentation + transforms_list
    
    transform = T.Compose(transforms_list)
    
    # Apply transformations
    img_tensor = transform(img)
    
    return img_tensor

# Function to load model
def load_model(settings):
    if settings["model_type"] == "resnet18":
        if settings["pretrained"]:
            weights = ResNet18_Weights.SENTINEL2_ALL
            model = resnet18(weights=weights)
        else:
            model = resnet18(in_channels=settings["in_channels"], num_classes=settings["num_classes"])
    
    model.eval()
    return model

# Function to run inference
def run_inference(model, img_tensor):
    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0))
    return outputs

# Function to visualize results
def visualize_results(img_tensor, outputs, model_type):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display original image
    img_np = img_tensor.permute(1, 2, 0).numpy()
    
    # If image was normalized, denormalize for display
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
    
    img_np = np.clip(img_np, 0, 1)
    ax1.imshow(img_np)
    ax1.set_title("Original Image")
    ax1.axis("off")
    
    # Display inference results based on model type
    if model_type == "resnet18":
        # If using pretrained model, show top predictions
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, 5)
        
        # For display purposes, create synthetic class names if we don't have actual class names
        class_names = [f"Class {i}" for i in range(len(probabilities))]
        
        # Plot top 5 predictions as a bar chart
        top_probs = top_probs.cpu().numpy()
        labels = [class_names[idx] for idx in top_indices.cpu().numpy()]
        
        ax2.barh(range(5), top_probs, color="skyblue")
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(labels)
        ax2.set_title("Top 5 Predictions")
        ax2.set_xlim(0, 1)
        
    plt.tight_layout()
    
    # Convert plot to image for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    
    return buf

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

with col2:
    st.header("Analysis Results")
    
    if uploaded_file is not None:
        # Create settings dictionary
        settings = {
            "model_type": model_type,
            "pretrained": pretrained,
            "in_channels": in_channels,
            "num_classes": num_classes,
            "resize_method": resize_method,
            "size": size,
            "normalize": normalize,
            "apply_augmentation": apply_augmentation
        }
        
        # Load model
        with st.spinner("Loading model..."):
            model = load_model(settings)
        
        # Process image and run inference
        with st.spinner("Processing image..."):
            img_tensor = preprocess_image(uploaded_file, settings)
            outputs = run_inference(model, img_tensor)
        
        # Visualize results
        result_img = visualize_results(img_tensor, outputs, model_type)
        st.image(result_img, caption="Analysis Results", use_column_width=True)
        
        # Show advanced details
        with st.expander("Advanced Details"):
            st.write(f"Model Type: {model_type}")
            st.write(f"Image Tensor Shape: {img_tensor.shape}")
            st.write(f"Output Shape: {outputs.shape}")
            
            # Display model architecture
            st.write("Model Architecture:")
            st.code(str(model))

# Display metadata about the analysis
if uploaded_file is not None:
    st.header("About this analysis")
    st.markdown("""
    This analysis uses TorchGeo, a PyTorch extension for geospatial data analysis. 
    The ResNet18 model is a common deep learning architecture for image classification.
    
    For remote sensing applications, these models can be used for:
    - Land cover classification
    - Object detection
    - Change detection
    - Anomaly detection
    
    Note: For production applications, you would typically fine-tune the model on your specific dataset.
    """)

# Footer
st.markdown("""
---
Created with Streamlit and TorchGeo
""")