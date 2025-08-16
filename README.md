# Satellite Image Classification Model Comparison

## Overview

This project provides a comprehensive toolkit for comparing different machine learning models on satellite image classification tasks. It supports analysis of Sentinel-2 imagery and includes multiple pre-trained models for comparison.

## Features

- **Multiple Model Support**: Compare predictions from various pre-trained models
- **Interactive Visualization**: Side-by-side comparison of model outputs
- **Flexible Input Formats**: Support for PNG, JPEG, TIFF, and direct URL inputs 
- **GPU Acceleration**: CUDA-enabled processing for faster inference
- **Export Options**: Save results in multiple formats (GeoTIFF, PNG)
- **Streamlit Interface**: User-friendly web interface for model interaction

## Installation

### Prerequisites

- Python 3.9+
- Conda package manager

### For Linux Users

```bash
# Create conda environment
conda env create -f [environment-linux.yml](http://_vscodecontentref_/1)

# Activate environment
conda activate classifier_comparison
```

### For Windows Users

```bash
conda env create -f [environment-windows.yml](http://_vscodecontentref_/2)

# Activate environment
conda activate classifier_comparison
```

For NVIDIA GPU users: Edit environment-windows.yml to include your CUDA version or install separately:

```bash
conda install pytorch torchvision cudatoolkit=<your_cuda_version> -c pytorch -c conda-forge
```

## Usage

### Running the Web Interface

```bash
streamlit run Home.py
```

### Using Jupyter Notebooks

Example notebooks are provided in the notebooks directory

- satellite_classification_demo.ipynb: Basic classification workflow
- example_use_cases.ipynb: Common use case examples
- user_interaction_flow.ipynb: Interface navigation guide

## Project Structure

```bash
├── data/                   # Sample data and test images
├── notebooks/             # Jupyter notebooks with examples
├── pages/                 # Streamlit interface pages
├── satellite_classifier/  # Core classification code
└── satellite_segmentation/# Segmentation utilities
```

## Contributing

Contributions are welcome ! Please feel free to submit a pull request.
