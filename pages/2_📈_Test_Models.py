import streamlit as st
import torch
import numpy as np
import rasterio
from rasterio.crs import CRS
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def select_sentinel2_bands(image, band_combination='rgb', num_bands=3):
    """
    Select and combine Sentinel-2 bands for visualization and classification.
    
    Args:
        image (numpy.ndarray): Input multi-band Sentinel-2 image
        band_combination (str): Band combination strategy ('rgb', 'false_color', 'vegetation')
        num_bands (int): Number of bands to select
    
    Returns:
        numpy.ndarray: Selected and potentially combined bands
    """
    # Common Sentinel-2 band indices (adjust based on your specific image)
    # Typical RGB-like combination: B2 (Blue), B3 (Green), B4 (Red)
    # Or false color: B8 (NIR), B4 (Red), B3 (Green)
    recommended_bands = {
        'rgb': [1, 2, 3],  # B2, B3, B4
        'false_color': [7, 3, 2],  # B8, B4, B3
        'vegetation': [7, 3, 1]  # B8, B4, B2
    }
    
    # Validate band selection
    total_bands = image.shape[2]
    if total_bands < num_bands:
        st.warning(f"Image has fewer bands ({total_bands}) than requested. Using all available bands.")
        return image
    
    # Select bands intelligently
    try:
        selected_band_indices = recommended_bands.get(band_combination, list(range(num_bands)))
        
        # Adjust indices to 0-based indexing
        selected_band_indices = [min(b-1, total_bands-1) for b in selected_band_indices[:num_bands]]
        
        # Select and return the chosen bands
        return image[:, :, selected_band_indices]
    except Exception as e:
        st.error(f"Error selecting bands: {e}")
        return image[:, :, :num_bands]

def visualize_rgb_image(image, band_combination='rgb'):
    """
    Create a visualized RGB image from Sentinel-2 data.
    
    Args:
        image (numpy.ndarray): Multi-band image
        band_combination (str): Band combination to use
    
    Returns:
        numpy.ndarray: 8-bit RGB image for visualization
    """
    # Select bands for visualization
    vis_image = select_sentinel2_bands(image, band_combination=band_combination)
    
    # Normalize each band to 0-1 range for visualization
    rgb_image = np.zeros_like(vis_image)
    for i in range(vis_image.shape[2]):
        band = vis_image[:,:,i]
        # Calculate percentiles for better contrast
        p_low, p_high = np.percentile(band[band > 0], (2, 98))
        
        # Clip and normalize
        rgb_image[:,:,i] = np.clip((band - p_low) / (p_high - p_low + 1e-10), 0, 1)
    
    # Convert to 8-bit
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # If this isn't in RGB order, reorder it
    if band_combination in ['false_color', 'vegetation']:
        # For visualization, we need to map NIR->R, R->G, etc.
        rgb_image = rgb_image[:, :, [0, 1, 2]]  # Ensure RGB display order
        
    return rgb_image

def preprocess_image(image, target_size=(224, 224), band_combination='rgb'):
    """
    Preprocess the Sentinel-2 image for classification.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Resize dimensions
        band_combination (str): Band combination to use
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    # Select and reduce bands
    image = select_sentinel2_bands(image, band_combination=band_combination, num_bands=3)
    
    # Convert to float and normalize
    image = image.astype(np.float32)
    
    # Normalize each band
    for i in range(image.shape[2]):
        band = image[:,:,i]
        mean = band.mean()
        std = band.std()
        
        # Avoid division by zero
        if std == 0:
            std = 1
        
        image[:,:,i] = (band - mean) / std
    
    # Resize image using OpenCV
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
    
    # Normalize across channels (optional, adjust as needed)
    image_tensor = (image_tensor - image_tensor.mean()) / image_tensor.std()
    
    # Add batch dimension
    return image_tensor.unsqueeze(0)

def load_sentinel2_image(image_path):
    """
    Load a Sentinel-2 image using rasterio with robust error handling.
    
    Args:
        image_path (str): Path to the Sentinel-2 image file
    
    Returns:
        tuple: (numpy.ndarray image, rasterio dataset metadata)
    """
    try:
        with rasterio.open(image_path) as src:
            # Read all bands
            image = src.read()
            
            # Handle potential CRS issues
            try:
                crs = src.crs
            except Exception as crs_error:
                st.warning(f"CRS Error: {crs_error}. Using default CRS.")
                crs = CRS.from_epsg(4326)  # Default to WGS84 if CRS fails
            
            # Transpose to get (Height, Width, Channels)
            image = np.transpose(image, (1, 2, 0))
            
            # Get metadata
            metadata = {
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'crs': crs,
                'bands': src.count
            }
            
            return image, metadata
    except rasterio.errors.RasterioError as e:
        st.error(f"Error loading image: {e}")
        return None, None

def load_pretrained_model(num_classes=7):
    """
    Load a pretrained model for classification.
    
    Args:
        num_classes (int): Number of classification classes
    
    Returns:
        torch.nn.Module: Pretrained classification model
    """
    try:
        import torchvision.models as models
        
        # Use ResNet18 with transfer learning
        model = models.resnet18(pretrained=True)
        
        # Modify the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, num_classes)
        
        return model
    except ImportError as e:
        st.error(f"Error loading model: {e}")
        return None

def classify_image(image_tensor, model, class_names):
    """
    Classify the preprocessed image.
    
    Args:
        image_tensor (torch.Tensor): Preprocessed image
        model (torch.nn.Module): Pretrained classification model
        class_names (list): List of class names
    
    Returns:
        tuple: Top prediction and its probability
    """
    if image_tensor is None or model is None:
        return "Classification Failed", 0.0
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient computation
    with torch.no_grad():
        # Get model predictions
        outputs = model(image_tensor)
        
        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top prediction
        top_prob, top_catid = probabilities.topk(1)
        
        # Convert to class name
        predicted_class = class_names[top_catid.item()]
        
        return predicted_class, top_prob.item()

def main():
    # Streamlit app setup
    st.set_page_config(
        page_title="Sentinel-2 Classifier",
        page_icon="🛰️",
        layout="wide"
    )
    st.title("Sentinel-2 Image Classifier")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a Sentinel-2 image", type=['tif', 'tiff'])
    
    if uploaded_file is not None:
        # Define class names (adjust based on your specific classification task)
        class_names = [
            'Forest', 'Agricultural', 'Urban', 'Water', 
            'Bare Soil', 'Grassland', 'Wetland'
        ]
        
        # Temporary save uploaded file
        with open("temp_sentinel2_image.tif", "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load the image
        image, metadata = load_sentinel2_image("temp_sentinel2_image.tif")
        
        if image is not None:
            # Create columns for image display and results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Image Visualization")
                
                # Display band combination options
                band_option = st.selectbox(
                    "Select Band Combination",
                    options=["RGB", "False Color", "Vegetation Index"],
                    index=0
                )
                
                # Map selection to band combination
                band_mapping = {
                    "RGB": "rgb",
                    "False Color": "false_color",
                    "Vegetation Index": "vegetation"
                }
                band_combination = band_mapping[band_option]
                
                # Create visualization
                rgb_image = visualize_rgb_image(image, band_combination=band_combination)
                
                # Display the image
                st.image(rgb_image, caption=f"Sentinel-2 Image ({band_option})", use_container_width=True)
                
                # Add options for image adjustments
                if st.checkbox("Show Image Enhancement Options"):
                    brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.1)
                    contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.1)
                    
                    # Apply adjustments
                    enhanced_image = cv2.convertScaleAbs(rgb_image, alpha=contrast, beta=brightness * 50)
                    st.image(enhanced_image, caption="Enhanced Image", use_container_width=True)
            
            with col2:
                st.subheader("Image Metadata")
                st.write(f"Image Shape: {image.shape}")
                st.write(f"Number of Bands: {metadata.get('bands', 'Unknown')}")
                st.write(f"CRS: {metadata.get('crs', 'Unknown')}")
                
                # Preprocess and classify
                st.subheader("Classification Results")
                
                with st.spinner("Classifying image..."):
                    # Preprocess the image
                    image_tensor = preprocess_image(image, band_combination=band_combination)
                    
                    # Load model
                    model = load_pretrained_model(num_classes=len(class_names))
                    
                    if image_tensor is not None and model is not None:
                        # Classify the image
                        predicted_class, confidence = classify_image(image_tensor, model, class_names)
                        
                        # Show results
                        st.markdown(f"**Predicted Class:** {predicted_class}")
                        st.markdown(f"**Confidence:** {confidence * 100:.2f}%")
                        
                        # Display confidence as a progress bar
                        st.progress(confidence)
                        
                        # Display classification information
                        st.info(f"Using band combination: {band_option}")
                    else:
                        st.error("Failed to preprocess image or load model")
        else:
            st.error("Failed to load image")

if __name__ == "__main__":
    main()

# Troubleshooting Requirements:
# pip install streamlit torch torchvision rasterio numpy opencv-python matplotlib pillow
#
# Key Features:
# 1. RGB visualization of Sentinel-2 imagery
# 2. Multiple band combination options (RGB, False Color, Vegetation)
# 3. Image enhancement controls (brightness/contrast)
# 4. Improved visualization for classification resultsS