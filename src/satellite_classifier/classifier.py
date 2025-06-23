import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO

from .constants import DEFAULT_CLASS_NAMES, PREPROCESSORS, IMAGENET_EARTH_OBS_INDICES

def _load_image(image_path_or_url):
    """
    Helper function to load an image from a local path or URL.
    """
    try:
        if image_path_or_url.startswith('http'):
            response = requests.get(image_path_or_url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(image_path_or_url).convert('RGB')
        print(f"✓ Loaded image: {img.size}")
        return img
    except Exception as e:
        print(f"✗ Error loading image: {e}")
        return None

def classify_satellite_image(models, image_path_or_url, class_names=None):
    """
    Classify a satellite image using multiple pretrained models.
    
    Args:
        models (dict): Dictionary of loaded models
        image_path_or_url (str): Path to local image or URL to satellite image
        class_names (dict, optional): Dictionary mapping model names to their class names.
                                     Defaults to DEFAULT_CLASS_NAMES.
        
    Returns:
        dict: Dictionary with model names as keys and prediction results as values
    """
    
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES
    
    img = _load_image(image_path_or_url)
    if img is None:
        return {}
    
    results = {}
    
    with torch.no_grad():
        for model_name, model in models.items():
            try:
                # Choose preprocessor and class names based on model type
                if 'ImageNet' in model_name:
                    preprocess = PREPROCESSORS['standard']
                    model_classes = class_names.get(model_name, DEFAULT_CLASS_NAMES['ResNet18_ImageNet'])
                    use_imagenet_filtering = True
                else:
                    preprocess = PREPROCESSORS['satellite']
                    model_classes = class_names.get(model_name, ['Unknown'] * 10) # Fallback if no specific classes
                    use_imagenet_filtering = False
                
                # Preprocess image
                img_tensor = preprocess(img).unsqueeze(0)
                
                # Forward pass
                outputs = model(img_tensor)
                
                # Handle different output formats (e.g., timm models sometimes have 'logits')
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=1)
                
                predictions = []
                
                if use_imagenet_filtering:
                    # For ImageNet models, filter to relevant earth observation classes only
                    for idx, class_name in IMAGENET_EARTH_OBS_INDICES.items():
                        if idx < probs.shape[1]:
                            prob = probs[0][idx].item()
                            predictions.append({
                                'class': class_name,
                                'confidence': prob,
                                'index': idx
                            })
                    predictions.sort(key=lambda x: x['confidence'], reverse=True)
                    predictions = predictions[:10] # Top 10 of the filtered classes
                else:
                    # For custom models, use all outputs and their corresponding class names
                    for i in range(min(len(model_classes), probs.shape[1])):
                        prob = probs[0][i].item()
                        predictions.append({
                            'class': model_classes[i],
                            'confidence': prob,
                            'index': i
                        })
                    predictions.sort(key=lambda x: x['confidence'], reverse=True)
                
                results[model_name] = {
                    'predictions': predictions,
                    'raw_output': logits.cpu().numpy(),
                    'probabilities': probs.cpu().numpy(),
                    'num_classes': len(model_classes)
                }
                
                print(f"✓ Classification completed for {model_name} ({len(model_classes)} classes)")
                
            except Exception as e:
                print(f"✗ Error with model {model_name}: {e}")
                results[model_name] = {'error': str(e)}
    
    return results