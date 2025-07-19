import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO

def _load_image(image_path_or_url):
    """
    Helper function to load an image from a local path or URL.
    
    Args:
        image_path_or_url (str): Path to local image or URL to satellite image
    """
    try:
        if image_path_or_url.startswith('http'):
            response = requests.get(image_path_or_url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(image_path_or_url).convert('RGB')
        return img
    except Exception as e:
        print(f"✗ Error loading image for visualization: {e}")
        return None

def visualize_comparison(image_path_or_url, results, title="Satellite Image Classification Comparison"):
    """
    Visualize classification results from multiple models side by side.
    
    Args:
        image_path_or_url (str): Path to the classified image
        results (dict): Results from classify_satellite_image function
        title (str): Title for the visualization
    """
    img = _load_image(image_path_or_url)
    if img is None:
        return
    
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    n_models = len(valid_results)
    
    if n_models == 0:
        print("✗ No valid results to visualize")
        return
    
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 6))
    
    # Handle case where only one model result exists (axes is not an array)
    if n_models == 0: # This case is already handled above, but as a safeguard
        axes = [axes]
    elif n_models == 1:
        axes = [axes[0], axes[1]] # Make sure axes is always iterable for consistent indexing
    
    # Display original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Display results for each model
    plot_idx = 1
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#B6CCFE'] # Added one more color
    
    for i, (model_name, result) in enumerate(valid_results.items()):
        ax = axes[plot_idx]
        
        # Get top 5 predictions
        predictions = result['predictions'][:5]
        classes = [p['class'][:20] + '...' if len(p['class']) > 20 else p['class'] 
                  for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(classes))
        bars = ax.barh(y_pos, confidences, 
                      color=colors[i % len(colors)], alpha=0.8)
        
        # Customize chart
        ax.set_yticks(y_pos)
        ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlabel('Confidence Score', fontsize=11)
        ax.set_title(f'{model_name}\nTop 5 Predictions', 
                    fontsize=12, fontweight='bold')
        ax.set_xlim(0, max(confidences) * 1.1)
        
        # Add confidence values on bars
        for j, (bar, conf) in enumerate(zip(bars, confidences)):
            ax.text(conf + max(confidences) * 0.02, 
                   bar.get_y() + bar.get_height()/2, 
                   f'{conf:.3f}', va='center', fontsize=9, fontweight='bold')
        
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('#F8F9FA')
        
        plot_idx += 1
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()