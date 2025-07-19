from .models import load_pretrained_models
from .classifier import classify_satellite_image
from .visualization import visualize_comparison
from .export import export_results
from .constants import DEFAULT_CLASS_NAMES, PREPROCESSORS, IMAGENET_EARTH_OBS_INDICES

"""
Satellite Image Classification Model Comparison Utilities

This package provides functions to load pretrained models,
perform classification on satellite imagery, and visualize results.
Avoids problematic TorchGeo imports by using direct model loading.
"""

__all__ = [
    "load_pretrained_models",
    "classify_satellite_image",
    "visualize_comparison",
    "export_results",
    "DEFAULT_CLASS_NAMES",
    "PREPROCESSORS",
    "IMAGENET_EARTH_OBS_INDICES"
]