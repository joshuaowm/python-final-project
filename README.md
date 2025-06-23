# Python Final Project: Satellite Image Classification Model Comparison

## Overview

This project provides utilities for classifying satellite images using various pre-trained models.

## Features

* **Sentinel-2 Imagery Support**: Work with Level 1C products, including various spectral bands.
* **Model Comparison**: Compare the performance of different classification models on satellite imagery.
* **Modular Design**: Organized codebase for easy understanding and extension.

## Setup Instructions

To get started, you'll need to set up a Conda environment with the necessary dependencies. Please choose the appropriate setup based on your operating system:

### For Linux Users

1. Navigate to the project root directory in your terminal.
2. Create the Conda environment using the Linux-specific file:

    ```bash
    conda env create -f environment-linux.yml
    ```

3. Activate the environment:

    ```bash
    conda activate classifier_comparison
    ```

### For Windows Users

1. Navigate to the project root directory in your Anaconda Prompt or terminal.
2. Create the Conda environment using the Windows-specific file:

    ```bash
    conda env create -f environment-windows.yml
    ```

3. Activate the environment:

    ```bash
    conda activate classifier_comparison
    ```

4. (Optional: If you have an NVIDIA GPU) Remember to edit `environment-windows.yml` to replace `cpuonly` with your specific `cudatoolkit` version *before* creating the environment, or install it afterward:

    ```bash
      conda install pytorch torchvision cudatoolkit=<your_cuda_version> -c pytorch -c conda-forge
    ```

## Running the Notebooks

Once your environment is active, you can launch Jupyter Lab/Notebook:

```bash
jupyter notebook
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
