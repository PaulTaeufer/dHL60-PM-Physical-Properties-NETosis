import os
import numpy as np
import tifffile
from skimage import filters, morphology, measure, draw, exposure
from scipy import ndimage
import matplotlib.pyplot as plt
import pandas as pd

# Function to create a nucleus mask
def create_nucleus_mask(dna_image, min_mask_size=100):
    # Smooth the image using a Gaussian filter
    dna_smoothed = filters.gaussian(dna_image, sigma=1)
    
    # Subtract background (optional)
    background = ndimage.minimum_filter(dna_smoothed, size=50)
    dna_subtracted = dna_smoothed - background
    
    # Apply Otsu's thresholding
    thresh = filters.threshold_otsu(dna_subtracted)
    binary_mask = dna_subtracted > thresh
    
    # Remove small objects
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_mask_size)
    
    # Fill holes
    filled_mask = ndimage.binary_fill_holes(cleaned_mask)
    
    return filled_mask.astype(np.uint8) * 255

# Function to calculate roundness
def calculate_roundness(mask):
    # Label the mask
    labeled_mask = measure.label(mask)
    
    # Calculate properties of the labeled regions
    regions = measure.regionprops(labeled_mask)
    
    # If no regions are found, return NaN
    if len(regions) == 0:
        return np.nan
    
    # Assume the largest region is the nucleus
    largest_region = max(regions, key=lambda x: x.area)
    
    # Calculate roundness
    perimeter = largest_region.perimeter
    area = largest_region.area
    if perimeter == 0:
        return np.nan
    roundness = (4 * np.pi * area) / (perimeter ** 2)
    
    return roundness

# Function to calculate mean and std intensity within the mask
def calculate_intensity_stats(image, mask):
    # Ensure the mask is binary
    mask = mask > 0
    
    # If the mask is empty, return NaN
    if np.sum(mask) == 0:
        return np.nan, np.nan
    
    # Calculate mean and std intensity within the mask
    mean_intensity = np.mean(image[mask])
    std_intensity = np.std(image[mask])
    
    return mean_intensity, std_intensity

# Function to calculate the area of the mask
def calculate_mask_area(mask):
    # Ensure the mask is binary
    mask = mask > 0
    
    # Calculate the area (number of pixels in the mask)
    area = np.sum(mask)
    
    return area

# Function to create a composite image with marker in green and mask outlines
def create_marker_overlay(marker_image, mask):
    # Normalize the marker image to the range [0, 1]
    marker_normalized = exposure.rescale_intensity(marker_image, out_range=(0, 1))
    
    # Create an RGB image
    overlay_image = np.zeros((marker_image.shape[0], marker_image.shape[1], 3), dtype=np.float32)
    
    # Assign the marker channel to the green channel
    overlay_image[..., 1] = marker_normalized  # Green channel (Marker)
    
    # Overlay mask outlines in red
    contours = measure.find_contours(mask, 0.5)
    for contour in contours:
        rr, cc = draw.polygon_perimeter(contour[:, 0], contour[:, 1], shape=mask.shape)
        overlay_image[rr, cc, 0] = 1.0  # Red channel (outline)
    
    return overlay_image

# Function to process a single TIFF file
def process_tiff_file(tiff_path, output_folder, min_mask_size=100):
    # Create a subfolder for this cell
    base_name = os.path.splitext(os.path.basename(tiff_path))[0]
    cell_output_folder = os.path.join(output_folder, base_name)
    os.makedirs(cell_output_folder, exist_ok=True)
    
    # Load the multi-dimensional TIFF file
    with tifffile.TiffFile(tiff_path) as tif:
        image_stack = tif.asarray()  # Load the entire stack into a numpy array
    
    # Check the shape of the image stack
    print(f"Processing file: {tiff_path}")
    print(f"Image stack shape: {image_stack.shape}")
    # Expected shape: (time_points, z_positions, channels, height, width)
    # If the shape is different, adjust the indexing accordingly
    
    # Extract dimensions
    num_time_points = image_stack.shape[0]
    num_z_positions = image_stack.shape[1]
    num_channels = image_stack.shape[2]
    
    # Ensure the DIC, DNA, and marker channels are present
    if num_channels < 3:
        raise ValueError("The TIFF file does not contain enough channels. Expected at least 3 channels.")
    
    # Create empty arrays to store masks and overlays for each z-position
    mask_stack_z0 = np.zeros((num_time_points, image_stack.shape[3], image_stack.shape[4]), dtype=np.uint8)
    mask_stack_z1 = np.zeros((num_time_points, image_stack.shape[3], image_stack.shape[4]), dtype=np.uint8)
    
    overlay_stack_z0 = np.zeros((num_time_points, image_stack.shape[3], image_stack.shape[4], 3), dtype=np.float32)  # For marker + mask overlays (z0)
    overlay_stack_z1 = np.zeros((num_time_points, image_stack.shape[3], image_stack.shape[4], 3), dtype=np.float32)  # For marker + mask overlays (z1)
    
    # Create lists to store roundness, mean intensity, std intensity, area, and marker intensity stats
    time_points = np.arange(num_time_points)
    data = {
        "Cell": [],
        "Time Point": [],
        "Z-Position": [],
        "Roundness": [],
        "Mean Intensity": [],
        "Std Intensity": [],
        "Area": [],
        "Marker Mean Intensity": [],
        "Marker Std Intensity": []
    }
    
    # Loop through all time points and z-positions
    for time_point in range(num_time_points):
        for z_position in range(num_z_positions):
            print(f"Processing time point {time_point}, z-position {z_position}...")
            
            # Extract the DIC, DNA, and marker channels
            dic_image = image_stack[time_point, z_position, 0, :, :]  # First channel (DIC)
            dna_image = image_stack[time_point, z_position, 1, :, :]  # Second channel (DNA)
            marker_image = image_stack[time_point, z_position, 2, :, :]  # Third channel (Marker)
            
            # Create the nucleus mask
            mask = create_nucleus_mask(dna_image, min_mask_size=min_mask_size)
            
            # Store the mask in the appropriate stack based on z-position
            if z_position == 0:
                mask_stack_z0[time_point, :, :] = mask
            elif z_position == 1:
                mask_stack_z1[time_point, :, :] = mask
            
            # Calculate roundness
            roundness = calculate_roundness(mask)
            
            # Calculate mean and std intensity for DNA channel
            mean_intensity, std_intensity = calculate_intensity_stats(dna_image, mask)
            
            # Calculate area
            area = calculate_mask_area(mask)
            
            # Calculate mean and std intensity for marker channel
            marker_mean_intensity, marker_std_intensity = calculate_intensity_stats(marker_image, mask)
            
            # Append data to the dictionary
            data["Cell"].append(base_name)
            data["Time Point"].append(time_point)
            data["Z-Position"].append(z_position)
            data["Roundness"].append(roundness)
            data["Mean Intensity"].append(mean_intensity)
            data["Std Intensity"].append(std_intensity)
            data["Area"].append(area)
            data["Marker Mean Intensity"].append(marker_mean_intensity)
            data["Marker Std Intensity"].append(marker_std_intensity)
            
            # Create a composite image with marker in green and mask outlines
            overlay_image = create_marker_overlay(marker_image, mask)
            
            # Store the overlay image in the appropriate stack based on z-position
            if z_position == 0:
                overlay_stack_z0[time_point, :, :, :] = overlay_image
            elif z_position == 1:
                overlay_stack_z1[time_point, :, :, :] = overlay_image
    
    # Save the mask stacks as separate TIFF files
    mask_file_z0 = os.path.join(cell_output_folder, f"{base_name}_nucleus_masks_z0.tif")
    mask_file_z1 = os.path.join(cell_output_folder, f"{base_name}_nucleus_masks_z1.tif")
    tifffile.imsave(mask_file_z0, mask_stack_z0)
    tifffile.imsave(mask_file_z1, mask_stack_z1)
    
    # Save the overlay stacks as separate TIFF files
    overlay_file_z0 = os.path.join(cell_output_folder, f"{base_name}_marker_overlays_z0.tif")
    overlay_file_z1 = os.path.join(cell_output_folder, f"{base_name}_marker_overlays_z1.tif")
    tifffile.imsave(overlay_file_z0, (overlay_stack_z0 * 255).astype(np.uint8))  # Scale to 8-bit
    tifffile.imsave(overlay_file_z1, (overlay_stack_z1 * 255).astype(np.uint8))  # Scale to 8-bit
    
    print(f"Masks for z-position 0 saved to {mask_file_z0}")
    print(f"Masks for z-position 1 saved to {mask_file_z1}")
    print(f"Marker overlays for z-position 0 saved to {overlay_file_z0}")
    print(f"Marker overlays for z-position 1 saved to {overlay_file_z1}")
    
    # Plot roundness, intensity, area, and marker intensity for this cell
    plot_cell_data(data, cell_output_folder, base_name)
    
    # Return the data for this cell
    return data

# Function to plot roundness, intensity, area, and marker intensity for a single cell
def plot_cell_data(data, output_folder, base_name):
    # Convert data to a DataFrame
    df = pd.DataFrame(data)
    
    # Plot roundness over time
    plt.figure(figsize=(10, 6))
    for z_pos in df["Z-Position"].unique():
        z_data = df[df["Z-Position"] == z_pos]
        plt.plot(z_data["Time Point"], z_data["Roundness"], label=f"Z{z_pos}", marker="o", markersize=4)
    plt.xlabel("Time Point")
    plt.ylabel("Roundness")
    plt.title(f"Roundness of Nucleus Masks Over Time ({base_name})")
    plt.legend()
    plt.grid(True)
    
    # Save the roundness plot
    roundness_plot_file = os.path.join(output_folder, f"{base_name}_roundness_plot.png")
    plt.savefig(roundness_plot_file)
    print(f"Roundness plot saved to {roundness_plot_file}")
    plt.close()
    
    # Plot mean and std intensity over time
    plt.figure(figsize=(10, 6))
    for z_pos in df["Z-Position"].unique():
        z_data = df[df["Z-Position"] == z_pos]
        plt.plot(z_data["Time Point"], z_data["Mean Intensity"], label=f"Mean (Z{z_pos})", marker="o", markersize=4)
        plt.plot(z_data["Time Point"], z_data["Std Intensity"], label=f"Std (Z{z_pos})", marker="o", markersize=4, linestyle="--")
    plt.xlabel("Time Point")
    plt.ylabel("Intensity")
    plt.title(f"Mean and Std Intensity Within Masks Over Time ({base_name})")
    plt.legend()
    plt.grid(True)
    
    # Save the intensity plot
    intensity_plot_file = os.path.join(output_folder, f"{base_name}_intensity_plot.png")
    plt.savefig(intensity_plot_file)
    print(f"Intensity plot saved to {intensity_plot_file}")
    plt.close()
    
    # Plot area over time
    plt.figure(figsize=(10, 6))
    for z_pos in df["Z-Position"].unique():
        z_data = df[df["Z-Position"] == z_pos]
        plt.plot(z_data["Time Point"], z_data["Area"], label=f"Z{z_pos}", marker="o", markersize=4)
    plt.xlabel("Time Point")
    plt.ylabel("Area (pixels)")
    plt.title(f"Area of Nucleus Masks Over Time ({base_name})")
    plt.legend()
    plt.grid(True)
    
    # Save the area plot
    area_plot_file = os.path.join(output_folder, f"{base_name}_area_plot.png")
    plt.savefig(area_plot_file)
    print(f"Area plot saved to {area_plot_file}")
    plt.close()
    
    # Plot marker mean and std intensity over time
    plt.figure(figsize=(10, 6))
    for z_pos in df["Z-Position"].unique():
        z_data = df[df["Z-Position"] == z_pos]
        plt.plot(z_data["Time Point"], z_data["Marker Mean Intensity"], label=f"Mean (Z{z_pos})", marker="o", markersize=4)
        plt.plot(z_data["Time Point"], z_data["Marker Std Intensity"], label=f"Std (Z{z_pos})", marker="o", markersize=4, linestyle="--")
    plt.xlabel("Time Point")
    plt.ylabel("Marker Intensity")
    plt.title(f"Marker Intensity Within Masks Over Time ({base_name})")
    plt.legend()
    plt.grid(True)
    
    # Save the marker intensity plot
    marker_intensity_plot_file = os.path.join(output_folder, f"{base_name}_marker_intensity_plot.png")
    plt.savefig(marker_intensity_plot_file)
    print(f"Marker intensity plot saved to {marker_intensity_plot_file}")
    plt.close()

# Main function to process all TIFF files in a folder
def process_folder(input_folder, output_folder, min_mask_size=100):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Initialize a list to store data from all cells
    all_data = []
    
    # Loop through all TIFF files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            tiff_path = os.path.join(input_folder, filename)
            cell_data = process_tiff_file(tiff_path, output_folder, min_mask_size=min_mask_size)
            all_data.append(cell_data)
    
    # Combine data from all cells into a single DataFrame
    combined_data = {key: [] for key in all_data[0].keys()}
    for cell_data in all_data:
        for key in cell_data:
            combined_data[key].extend(cell_data[key])
    
    df = pd.DataFrame(combined_data)
    
    # Save the combined data to a single CSV file
    csv_file = os.path.join(output_folder, "all_cells_data.csv")
    df.to_csv(csv_file, index=False)
    print(f"Combined data saved to {csv_file}")

# Parameters
input_folder = "cells"  # Folder containing TIFF files
output_folder = "out"   # Output directory
min_mask_size = 100     # Minimum size of masks to consider (in pixels)

# Run the processing
process_folder(input_folder, output_folder, min_mask_size=min_mask_size)