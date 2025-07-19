from datetime import datetime
import os
import json
import numpy as np

def _json_serializable(obj):
    """Convert numpy arrays to JSON serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def convert_raw_output(results):
    """Convert numpy arrays in 'raw_output' keys to lists."""
    if not isinstance(results, dict):
        return results
    
    for model_name, result in results.items():
        if 'raw_output' in result:
            if isinstance(result['raw_output'], np.ndarray):
                result['raw_output'] = result['raw_output'].tolist()
    
    return results

def export_results(results=None, mode='text', output_dir='../results'):
    """Export classification results to a file.
    Args:
        results (dict, optional): The classification results to export. Defaults to None.
        mode (str, optional): The mode of export ('text' or 'json'). Defaults to 'text'.
        output_dir (str, optional): The directory to save the results. Defaults to '../results'.
    """
    # Get the timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
   
    # Define the output file path
    if mode == 'text':
        output_file = os.path.join(output_dir, f'classification_results_{timestamp}.txt')
    elif mode == 'json':
        output_file = os.path.join(output_dir, f'classification_results_{timestamp}.json')
    else:
        raise ValueError("Unsupported export mode. Use 'text' or 'json'.")
   
    if results is None or not results or not isinstance(results, dict):
        # If results are empty or not a dictionary, we cannot export
        print("✗ No results to export.")
        results = {"message": "No results to export."}
    
    try:
        if mode == 'text':
            with open(output_file, 'w') as f:
                for model_name, result in results.items():
                    f.write(f"Model: {model_name}\n")
                    if isinstance(result, dict):
                        for key, value in result.items():
                            f.write(f"{key}: {value}\n")
                    else:
                        f.write(f"Result: {result}\n")
                    f.write("\n")
            print(f"✓ Results exported to {output_file}")
        elif mode == 'json':
            # Convert numpy arrays only in 'raw_output' keys
            serializable_results = convert_raw_output(results.copy())
            with open(output_file, 'w') as f:
                json.dump(serializable_results, f, indent=4, default=_json_serializable)
            print(f"✓ Results exported to {output_file}")
    except Exception as e:
        print(f"✗ Error exporting results: {e}")
   
    return output_file