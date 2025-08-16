"""Constants for the satellite segmentation module"""

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
    "UperNet Models": {
        "upernet-swin-base": {
            "model_name": "openmmlab/upernet-swin-base",
            "description": "UperNet-Swin-Base ADE20K (150 classes)",
            "type": "transformers"
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