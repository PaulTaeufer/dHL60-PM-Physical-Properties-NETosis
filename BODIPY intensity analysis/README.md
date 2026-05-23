# BODIPY Intensity Analysis

## Overview
This folder contains code and analysis scripts for quantifying BODIPY fluorescence intensity in dHL60-PM cells during NETosis (Neutrophil Extracellular Trap formation). BODIPY is used as a marker for lipid content and cellular properties during programmed cell death.

## Purpose
The analysis:
- quantifies BODIPY fluorescence intensity in cells
- Visualization of intensity distributions and temporal dynamics

## Code Structure

### Main Scripts
- **Image Processing**: Loads and processes fluorescence microscopy images
- **Intensity Quantification**: Extracts BODIPY signal from regions of interest (ROIs)
- **Statistical Analysis**: Performs statistical tests and comparisons
- **Visualization**: Generates plots and graphs for data interpretation

## Key Features
- Automated ROI detection and signal extraction
- Background correction
- Normalization of intensity values
- Temporal analysis of intensity changes
- Statistical comparison between control and treatment groups

## Data Input
- Fluorescence microscopy images (TIFF, CZI, or other microscopy formats)
- Experimental metadata (timepoints, treatment conditions)
- Cell segmentation masks (if available)

## Data Output
- Quantified BODIPY intensity values per cell/ROI
- Statistical summary tables
- Visualization plots (histograms, time-series, scatter plots)
- Excel or CSV formatted results

## Dependencies
Common libraries used:
- NumPy/SciPy: Numerical analysis
- Matplotlib/Seaborn: Visualization
- scikit-image: Image processing
- pandas: Data manipulation
- PIL/OpenCV: Image handling

## Usage
Refer to individual script headers and comments for specific execution instructions. Typically:
1. Place raw microscopy images in the input folder
2. Run the main analysis script
3. Check output folder for results

## Author Notes
This analysis was conducted as part of the dHL60-PM Physical Properties during NETosis research project.
