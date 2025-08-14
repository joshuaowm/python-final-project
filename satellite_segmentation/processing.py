"""Image processing utilities"""
import numpy as np

def create_color_palette(num_classes=150):
    """Create color palette for visualization"""
    np.random.seed(42)
    palette = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # Background as black
    return palette