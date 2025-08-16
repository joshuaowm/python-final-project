"""
Semantic Segmentation Web App using Streamlit
Simplified modular architecture
"""

import streamlit as st
import numpy as np
from PIL import Image
import torch

from satellite_segmentation.constants import get_available_models
from satellite_segmentation.models import load_model, process_image, is_binary_mask
from satellite_segmentation.visualization import create_visualization, create_class_distribution_chart

# Page config
st.set_page_config(
    page_title="Semantic Segmentation Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🎯 Semantic Segmentation Hub")
    st.markdown("Upload an image and choose a model to perform semantic segmentation")
    
    # Get available models
    available_models = get_available_models()
    
    # Sidebar
    st.sidebar.header("Model Selection")
    
    # Flatten model options for selectbox
    model_options = []
    model_mapping = {}
    
    for category, models in available_models.items():
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
                    # Load model
                    model, processor, device = load_model(selected_config)
                    
                    if model is None:
                        st.error("Failed to load model")
                        return
                
                with st.spinner("Running inference..."):
                    # Process image
                    model_type = selected_config["type"]
                    mask, inference_time = process_image(image, model, processor, device, model_type)
                    binary_mask = is_binary_mask(model_type)
                
                # Display results
                st.success(f"Segmentation completed in {inference_time:.2f} seconds")
                
                # Create and display visualization
                fig = create_visualization(image, mask, selected_config['description'], binary_mask)
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
                    if binary_mask:
                        segmented_pixels = np.sum(mask == 1)
                        total_pixels = mask.size
                        percentage = (segmented_pixels / total_pixels) * 100
                        st.metric("Segmented Area", f"{percentage:.1f}%")
                    else:
                        st.metric("Mask Shape", f"{mask.shape[0]}×{mask.shape[1]}")
                
                # Class distribution (for multi-class models)
                if not binary_mask and unique_classes > 1:
                    st.header("Class Distribution")
                    fig = create_class_distribution_chart(mask)
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
        - **UperNet**: Unified Perceptual Parsing for Scene Understanding
        - **SMP**: Segmentation Models PyTorch library models
        """)
    with c3:
        st.markdown("""
        ### Tips:
        - Different models are trained on different datasets (Cityscapes, ADE20K)
        - UperNet combines multiple scales for detailed scene understanding
        - GPU acceleration will be used if available
        """)

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and PyTorch")

if __name__ == "__main__":
    main()