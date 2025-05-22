# Python Final Project: Web GIS Processing

## Overview

This project focuses on processing Sentinel-2 Level 1C satellite imagery using Python. It aims to provide tools and scripts for analyzing and visualizing geospatial data, facilitating tasks such as:

* Reading and handling Sentinel-2 imagery
* Performing basic image processing operations
* Visualizing different spectral bands
* Preparing data for further geospatial analysis

## Features

* **Sentinel-2 Imagery Support**: Work with Level 1C products, including various spectral bands.
* **Image Processing**: Apply operations like band extraction and visualization.
* **Modular Design**: Organized codebase for easy understanding and extension.

## Getting Started

### Prerequisites

* Python 3.7 or higher
* Recommended to use a virtual environment

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/python-amazing/python-final-project.git
   cd python-final-project
   ```

2. **Set up the environment**:

   It's recommended to use a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   The project uses a `requirements.yaml` file. If this is a conda environment file, you can create the environment using:

   ```bash
   conda env create -f requirements.yaml
   ```

   If it's a pip requirements file, rename it to `requirements.txt` and install using:

   ```bash
   pip install -r requirements.txt
   ```

   *(Note: Please verify the format of `requirements.yaml` to ensure correct installation.)*

## Usage

1. **Run the main script**:

   ```bash
   python 1__Home.py
   ```

   This script serves as the entry point to the application.

2. **Input Data**:

   Ensure that the Sentinel-2 `.tif` files are placed in the appropriate directory or update the script to point to their location.

3. **Output**:

   Processed images and results will be saved or displayed as per the script's functionality.

## Project Structure

```
├── 1__Home.py
├── Sentinel2_Lvl1C_BGRNIRSWIR1SWIR2_20150813.tif
├── temp_sentinel2_image.tif
├── requirements.yaml
├── LICENSE
└── README.md
```

* `1__Home.py`: Main script to execute the application.
* `Sentinel2_Lvl1C_BGRNIRSWIR1SWIR2_20150813.tif`: Sample Sentinel-2 imagery.
* `temp_sentinel2_image.tif`: Temporary file generated during processing.
* `requirements.yaml`: Dependencies required for the project.
* `LICENSE`: Project's license information.
* `README.md`: This file.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
