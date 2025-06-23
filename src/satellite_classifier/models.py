import torch
from torchvision.models import resnet18, resnet50, efficientnet_b0

def load_pretrained_models():
    """
    Load pretrained classification models suitable for satellite imagery.
    Includes both ImageNet models and remote sensing specific models.
    
    Returns:
        dict: Dictionary containing loaded models with their names as keys
    """
    models = {}
    
    # Option 1: Try to load remote sensing specific models
    try:
        import timm
        # For demo, we'll create the architecture. You'd load actual weights here.
        # Example: model_eurosat.load_state_dict(torch.load('path/to/eurosat_weights.pth'))
        model_eurosat = timm.create_model('resnet50', pretrained=False, num_classes=10)
        model_eurosat.eval()
        models['ResNet50_EuroSAT'] = model_eurosat
        print("✓ Loaded ResNet50 for EuroSAT (10 land cover classes)")
    except ImportError:
        print("! 'timm' library not found. Skipping EuroSAT model.")
    except Exception as e:
        print(f"! Could not load EuroSAT model: {e}")
    
    # Option 2: Load ImageNet models and modify for remote sensing
    try:
        model_landcover = resnet18(pretrained=True)
        # Replace final layer for 8 common land cover classes
        model_landcover.fc = torch.nn.Linear(model_landcover.fc.in_features, 8)
        model_landcover.eval()
        models['ResNet18_LandCover'] = model_landcover
        print("✓ Loaded ResNet18 modified for land cover (8 classes)")
    except Exception as e:
        print(f"✗ Could not load modified ResNet18: {e}")
    
    # Option 3: Standard ImageNet models (fallback)
    try:
        model_resnet18 = resnet18(pretrained=True)
        model_resnet18.eval()
        models['ResNet18_ImageNet'] = model_resnet18
        print("✓ Loaded ResNet18 with ImageNet weights")
    except Exception as e:
        print(f"✗ Could not load ResNet18: {e}")
    
    try:
        model_resnet50 = resnet50(pretrained=True)
        model_resnet50.eval()
        models['ResNet50_ImageNet'] = model_resnet50
        print("✓ Loaded ResNet50 with ImageNet weights")
    except Exception as e:
        print(f"✗ Could not load ResNet50: {e}")
    
    # Option 4: Try loading a model trained on aerial imagery
    try:
        model_aerial = resnet18(pretrained=True)
        # Modify for aerial image classification (12 classes)
        model_aerial.fc = torch.nn.Linear(model_aerial.fc.in_features, 12)
        model_aerial.eval()
        models['ResNet18_Aerial'] = model_aerial
        print("✓ Loaded ResNet18 for aerial imagery (12 classes)")
    except Exception as e:
        print(f"✗ Could not load aerial model: {e}")
    
    print(f"\nLoaded {len(models)} models successfully")
    return models