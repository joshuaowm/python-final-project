"""Model loading and processing functions"""
import time
import torch
import numpy as np
import streamlit as st
import albumentations as A

from transformers.models.auto.modeling_auto import AutoModelForSemanticSegmentation
from transformers.models.auto.image_processing_auto import AutoImageProcessor
from transformers.models.segformer import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)
from transformers.models.beit import (
    BeitImageProcessor,
    BeitForSemanticSegmentation,
    BeitFeatureExtractor
)
from transformers.models.dpt import (
    DPTImageProcessor,
    DPTForSemanticSegmentation
)
from transformers.models.sam import (
    SamProcessor,
    SamModel
)

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

@st.cache_resource
def load_transformers_model(model_name):
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

def load_model(model_config):
    """Load model based on config type"""
    model_name = model_config["model_name"]
    model_type = model_config["type"]
    
    if model_type == "transformers":
        return load_transformers_model(model_name)
    elif model_type == "sam":
        return load_sam_model(model_name)
    elif model_type == "smp":
        return load_smp_model(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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

def process_image(image, model, processor, device, model_type):
    """Process image based on model type"""
    if model_type == "transformers":
        return process_transformers_model(image, model, processor, device)
    elif model_type == "sam":
        return process_sam_model(image, model, processor, device)
    elif model_type == "smp":
        return process_smp_model(image, model, processor, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def is_binary_mask(model_type):
    """Check if model produces binary masks"""
    return model_type == "sam"