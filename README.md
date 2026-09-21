# 🛰️ Satellite Image Analysis Hub

## Overview

This project is an AI-powered platform for rapid satellite and aerial image analysis. It enables users to instantly classify and segment images using multiple state-of-the-art pre-trained models, providing quick and actionable insights.

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
conda env create -f environment-linux.yml

# Activate environment
conda activate classifier_comparison
```

### For Windows Users

```bash
conda env create -f environment-windows.yml

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

Example notebooks are provided in the `notebooks` directory:

- satellite_classification_demo.ipynb: Classification workflow
- semantic_segmentation_demo.ipynb: Segmentation workflow
- example_use_cases.ipynb: Common use case examples
- user_interaction_flow.ipynb: Interface navigation guide

## Project Structure

```
├── Home.py                   # Streamlit entry point
├── pages/                    # Streamlit pages (classification, segmentation)
├── satellite_classifier/     # Core classification code
├── satellite_segmentation/   # Segmentation utilities
├── notebooks/                # Jupyter notebooks with examples
├── data/                     # Sample data and test images
├── results/                  # Example outputs
├── test/                     # Segmentation outputs from the tested models
└── environment-*.yml         # Conda environments for Linux and Windows
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## Developers

**Bruna Cândido** <br>
LinkedIn: [Bruna Strack Cândido](https://www.linkedin.com/in/brunascandido/) <br>
Email: brunscandido@gmail.com

**Dastan Nurbekuly** <br>
LinkedIn: [Dastan Nurbekuly](https://www.linkedin.com/in/dastan-nurbekuly-1758362b3/) <br>
Email: dastan.nurbek22@gmail.com

**Elouann Lucas** <br>
LinkedIn: [Elouann Lucas](https://www.linkedin.com/in/elouann-lucas/) <br>
Email: elouannlucas56@gmail.com

**Joshua Mangotang** <br>
LinkedIn: [Joshua Owen Mangotang](https://www.linkedin.com/in/joshuaowm/) <br>
Email: joshuaowen1500@gmail.com

