import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import requests
from datetime import datetime
import tempfile
import os
import sys

try:
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
    from satellite_classifier import load_pretrained_models, classify_satellite_image, export_results
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #ebebe6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def format_confidence(confidence):
    """Format confidence score with color coding"""
    if confidence > 0.7:
        return f'<span class="confidence-high">{confidence:.3f}</span>'
    elif confidence > 0.3:
        return f'<span class="confidence-medium">{confidence:.3f}</span>'
    else:
        return f'<span class="confidence-low">{confidence:.3f}</span>'

import matplotlib.pyplot as plt
import numpy as np
import math

def create_results_visualization(image, results):
    """Create a matplotlib visualization of results with grid layout."""
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    n_models = len(valid_results)
    
    if n_models == 0:
        return None
    
    # Determine grid: 2 columns, enough rows for original + models
    n_cols = 2
    n_rows = math.ceil((n_models + 1) / n_cols)  # +1 for original image
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(-1)  # flatten to 1D for easy indexing
    
    # Display original image in first subplot
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Colors for different models
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#B6CCFE']
    
    # Display results for each model
    plot_idx = 1
    for i, (model_name, result) in enumerate(valid_results.items()):
        ax = axes[plot_idx]
        
        predictions = result['predictions'][:5]
        classes = [p['class'][:20] + '...' if len(p['class']) > 20 else p['class'] 
                  for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        y_pos = np.arange(len(classes))
        bars = ax.barh(y_pos, confidences, color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlabel('Confidence Score', fontsize=11)
        ax.set_title(f'{model_name}\nTop 5 Predictions', fontsize=12, fontweight='bold')
        ax.set_xlim(0, max(confidences) * 1.1 if confidences else 1)
        
        for bar, conf in zip(bars, confidences):
            ax.text(conf + (max(confidences) * 0.02 if confidences else 0.02),
                    bar.get_y() + bar.get_height()/2,
                    f'{conf:.3f}', va='center', fontsize=9, fontweight='bold')
        
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('#F8F9FA')
        plot_idx += 1
    
    # Hide unused subplots if any
    for ax in axes[plot_idx:]:
        ax.axis('off')
    
    # Main title
    fig.suptitle("Satellite Image Classification Results", fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

@st.cache_resource
def load_models():
    """Load models with caching"""
    with st.spinner("Loading classification models... This may take a moment."):
        return load_pretrained_models()

def main():
    st.title("🛰️ Satellite Image Classifier")
    st.markdown("Upload a satellite or aerial image to classify it using multiple deep learning models.")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    models = load_models()
    
    if not models:
        st.error("No models were loaded successfully. Please check the console for error messages.")
        return
    
    available_models = list(models.keys())
    selected_models = st.sidebar.multiselect(
        "Select models to use:",
        available_models,
        default=available_models
    )
    
    if not selected_models:
        st.warning("Please select at least one model.")
        return
    
    # Filter models based on selection
    filtered_models = {name: models[name] for name in selected_models}
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Image Input")
        
        # Image input options
        input_method = st.radio("Choose input method:", ["Upload Image", "Image URL"])
        
        image = None
        image_source = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
                help="Upload a satellite or aerial image for classification"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                image_source = uploaded_file.name
                
        else:  # URL input
            image_url = st.text_input(
                "Enter image URL:",
                placeholder="https://example.com/satellite_image.jpg"
            )
            
            if image_url:
                try:
                    with st.spinner("Loading image from URL..."):
                        response = requests.get(image_url)
                        image = Image.open(io.BytesIO(response.content)).convert('RGB')
                        image_source = image_url
                except Exception as e:
                    st.error(f"Error loading image from URL: {e}")
    
    with col2:
        # Display image
        if image:
            st.image(image, caption=f"Input Image: {image_source}", use_container_width=True)
            st.write(f"Image size: {image.size}")

    with col1:
        if image is not None:
            if st.button("🔍 Classify Image", type="primary"):
                with st.spinner("Classifying image... This may take a moment."):
                    # Save image temporarily for classification
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file.name)
                        tmp_path = tmp_file.name
                    
                    try:
                        # Perform classification
                        results = classify_satellite_image(filtered_models, tmp_path)
                        
                        # Store results in session state
                        st.session_state.results = results
                        st.session_state.image = image
                        st.session_state.image_source = image_source
                        
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_path)
        
        # Display results if available
        if hasattr(st.session_state, 'results') and st.session_state.results:
            results = st.session_state.results
            
            # Results summary
            st.markdown("### 📊 Classification Summary")
            
            valid_results = {k: v for k, v in results.items() if 'error' not in v}
            error_results = {k: v for k, v in results.items() if 'error' in v}
            
            if valid_results:
                st.success(f"Successfully classified with {len(valid_results)} model(s)")
            
            if error_results:
                st.warning(f"{len(error_results)} model(s) encountered errors")
                with st.expander("View errors"):
                    for model_name, result in error_results.items():
                        st.error(f"**{model_name}**: {result['error']}")
            
            # Detailed results for each model
            for model_name, result in valid_results.items():
                with st.expander(f"📈 {model_name} Results", expanded=True):
                    if 'predictions' in result:
                        predictions = result['predictions'][:10]  # Top 10
                        
                        st.write(f"**Number of classes:** {result.get('num_classes', 'Unknown')}")
                        
                        # Create a DataFrame for better display
                        import pandas as pd
                        pred_data = []
                        for i, pred in enumerate(predictions):
                            pred_data.append({
                                'Rank': i + 1,
                                'Class': pred['class'],
                                'Confidence': pred['confidence'],
                                'Index': pred['index']
                            })
                        
                        df = pd.DataFrame(pred_data)
                        
                        # Style the dataframe
                        def style_confidence(val):
                            if val > 0.7:
                                return 'color: #28a745; font-weight: bold'
                            elif val > 0.3:
                                return 'color: #ffc107; font-weight: bold'
                            else:
                                return 'color: #dc3545; font-weight: bold'
                        
                        styled_df = df.style.applymap(style_confidence, subset=['Confidence'])
                        styled_df = styled_df.format({'Confidence': '{:.4f}'})
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Top prediction highlight
                        if predictions:
                            top_pred = predictions[0]
                            confidence_html = format_confidence(top_pred['confidence'])
                            st.markdown(f"""
                            <div class="result-box">
                                <h4 style="color: #000">🏆 Top Prediction</h4>
                                <p style="color: #000"><strong>Class:</strong> {top_pred['class']}</p>
                                <p style="color: #000"><strong>Confidence:</strong> {confidence_html}</p>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Visualization section
    if hasattr(st.session_state, 'results') and hasattr(st.session_state, 'image'):
        st.markdown("---")
        st.subheader("📊 Visual Comparison")
        
        # Create and display visualization
        fig = create_results_visualization(st.session_state.image, st.session_state.results)
        if fig:
            st.pyplot(fig)
            plt.close(fig)  # Important to close the figure to prevent memory leaks
            
    # Export section
    if hasattr(st.session_state, 'results'):
        st.markdown("---")
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            export_format = st.selectbox(
                "Export format:",
                options=["text", "json"],
                help="Choose the format for exporting results"
            )
        
        with col2:
            if st.button("Export Results", type="primary"):
                try:
                    output_file = export_results(
                        results=st.session_state.results,
                        mode=export_format,
                        output_dir="results"
                    )
                    
                    # Create download button for the exported file
                    with open(output_file, 'r') as f:
                        file_content = f.read()
                        
                    filename = os.path.basename(output_file)
                    st.download_button(
                        label="📥 Download Results",
                        data=file_content,
                        file_name=filename,
                        mime="text/plain" if export_format == "text" else "application/json"
                    )
                    
                    st.success(f"Results exported successfully! Click the download button above to save the file.")
                except Exception as e:
                    st.error(f"Error exporting results: {e}")
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About the Models"):
        st.markdown("""
        This application uses multiple deep learning models for satellite image classification:
        
        **Model Types:**
        - **EuroSAT Models**: Specialized for European satellite imagery with 10 land cover classes
        - **Land Cover Models**: General land cover classification with 8 classes
        - **Aerial Models**: Aerial imagery classification with 12 classes  
        - **ImageNet Models**: General-purpose models adapted for satellite imagery
        
        **Usage Tips:**
        - Upload high-quality satellite or aerial images for best results
        - Different models may focus on different aspects of the image
        - Compare results across models to get a comprehensive understanding
        - Higher confidence scores (>0.7) indicate more reliable predictions
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and PyTorch")

if __name__ == "__main__":
    main()