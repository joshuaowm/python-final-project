"""Constants for the satellite segmentation module"""

AVAILABLE_MODELS = {
    "SegFormer Models": {
        "segformer-b5-cityscapes": {
            "model_name": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
            "description": "SegFormer-B5",
            "type": "transformers",
            "dataset": "Cityscapes (19 classes)"
        },
        "segformer-b2-ade": {
            "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",
            "description": "SegFormer-B2",
            "type": "transformers",
            "dataset": "ADE20K (150 classes)"
        },
    },
    "BEiT Models": {
        "beit-base-ade": {
            "model_name": "microsoft/beit-base-finetuned-ade-640-640",
            "description": "BEiT-Base",
            "type": "transformers",
            "dataset": "ADE20K (150 classes)"
        }
    },
    "DPT Models": {
        "dpt-large-ade": {
            "model_name": "Intel/dpt-large-ade",
            "description": "DPT-Large",
            "type": "transformers",
            "dataset": "ADE20K (150 classes)"
        }
    },
    "UperNet Models": {
        "upernet-swin-base": {
            "model_name": "openmmlab/upernet-swin-base",
            "description": "UperNet-Swin-Base",
            "type": "transformers",
            "dataset": "ADE20K (150 classes)"
        }
    }
}

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

def get_available_models():
    """Get available models based on installed packages"""
    models = AVAILABLE_MODELS.copy()
    
    if SMP_AVAILABLE:
        models["SMP Models"] = {
            "smp-segformer-city": {
                "model_name": "smp-hub/segformer-b2-1024x1024-city-160k",
                "description": "SMP SegFormer-B2 Cityscapes",
                "type": "smp"
            }
        }
    
    return models