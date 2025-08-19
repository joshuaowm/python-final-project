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
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🗺️ Semantic Segmentation Hub")
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
    st.sidebar.write(f"**Dataset:** {selected_config['dataset'].upper()}")
    
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
    
    st.markdown('<div style="height:25px;"></div>', unsafe_allow_html=True) # Gap
    
    # Information section
    with st.expander("🧠 About the Models"):
        st.markdown("""
    - **SegFormer-B5 (Cityscapes)** [🔗](https://huggingface.co/nvidia/segformer-b5-finetuned-cityscapes-1024-1024)<br> 
        A transformer-based semantic segmentation model from **NVIDIA**, using the large **B5 variant**.    
        Trained on the **Cityscapes dataset (19 classes)** for urban scene understanding.  

    - **SegFormer-B2 (ADE20K)** [🔗](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)<br>
        A medium-sized **B2 variant** of NVIDIA’s SegFormer.    
        Trained on **ADE20K (150 classes)**, suitable for broad scene parsing.    

    - **BEiT-Base (ADE20K)** [🔗](https://huggingface.co/microsoft/beit-base-finetuned-ade-640-640)<br>
        A **BERT-style image transformer** by **Microsoft**, adapted for segmentation.  
        Fine-tuned on **ADE20K (150 classes)** for dense semantic segmentation.    

    - **DPT-Large (ADE20K)** [🔗](https://huggingface.co/Intel/dpt-large-ade)<br>
        The **large variant** of the Dense Prediction Transformer (DPT) by **Intel**.  
        Pre-trained and fine-tuned on **ADE20K (150 classes)** for segmentation and other dense tasks.    

    - **UperNet-Swin-Base (ADE20K)** [🔗](https://huggingface.co/openmmlab/upernet-swin-base)<br>
        A **UperNet architecture** using a **Swin Transformer-Base** backbone, from **OpenMMLab**.  
        Trained on **ADE20K (150 classes)** for comprehensive scene parsing.    

    - **SMP (Segmentation Models PyTorch)** [🔗](https://github.com/qubvel/segmentation_models.pytorch)<br>
        A library offering multiple architectures (**U-Net, FPN, DeepLabV3+**, etc.) with various backbones.  
        Widely used for training custom segmentation pipelines.    
    """, unsafe_allow_html=True)


        
    with st.expander("ℹ️ Usage Tips"):
        st.markdown("""
        - Different models are trained on different datasets (Cityscapes, ADE20K)
        - UperNet combines multiple scales for detailed scene understanding
        - GPU acceleration will automatically be used if available
        """)

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, 🔥PyTorch and 🤗Transformers")

if __name__ == "__main__":
    main()