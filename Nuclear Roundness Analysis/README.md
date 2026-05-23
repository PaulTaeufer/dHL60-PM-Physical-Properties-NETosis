# Nuclear Roundness Analysis

## Overview

This directory contains the `Nucleus_Masker.py` script, which automates the analysis of nuclear morphology, specifically focusing on measuring the roundness of cell nuclei in microscopy images.

## Nucleus_Masker.py

### Purpose

`Nucleus_Masker.py` is designed to:
- Segment and identify individual nuclei in fluorescence microscopy images
- Generate binary masks for nuclear regions
- Calculate morphological properties of nuclei, including roundness measurements
- Process batch images from a specified directory
- Visualize nuclear masks for quality control and validation

### Key Features

- **Automated nuclei detection**: Uses image processing techniques to identify nuclear boundaries
- **Morphological analysis**: Computes shape descriptors including:
  - Roundness (circularity)
  - Area
  - Perimeter
  - Solidity
  - Eccentricity
- **Batch processing**: Processes multiple images in sequence
- **Visualization output**: Generates overlay images showing detected nuclei
- **Data export**: Saves morphological measurements for further statistical analysis

### Input Requirements

- Fluorescence microscopy images (supported formats: TIFF, PNG, JPG)
- Images should contain DAPI or similar nuclear staining
- Image files organized in an input directory

### Output Files

- **Binary masks**: Nuclear segmentation masks (per image)
- **Measurements CSV**: Quantitative morphological data
- **Visualization images**: Overlay images with detected nuclei highlighted
- **Summary statistics**: Overall analysis results and parameters used

### Usage

Basic usage:
```bash
python Nucleus_Masker.py --input_dir <path_to_images> --output_dir <output_path>
```

### Parameters

Configure the following in the script or via command-line arguments:
- Threshold values for nuclear detection
- Minimum and maximum nucleus size filters
- Morphological operations (erosion, dilation)
- Output image resolution and format

### Dependencies

- OpenCV (cv2)
- NumPy
- scikit-image
- Pandas
- Matplotlib

### Algorithm Overview

1. **Image preprocessing**: Contrast enhancement and noise reduction
2. **Segmentation**: Binary thresholding or adaptive methods
3. **Morphological operations**: Cleaning and refining nuclear masks
4. **Connected component analysis**: Labeling individual nuclei
5. **Feature extraction**: Computing roundness and other shape descriptors
6. **Filtering**: Removing artifacts based on size and shape criteria
7. **Quantification**: Exporting measurements and visualizations
