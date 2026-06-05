# Intensity Normalization Tool
# This script processes the Excel file 'Mask-Cholesterol-MeanInt_data.xlsx' located
# in the same folder. It works with the 'Raw' sheet, where every three columns represent
# data from a single cell: time, state, and intensity.
# Functionality:
# - Extracts intensity values for each cell.
# - Normalizes each intensity series by the value at the FIRST occurrence of a given NETosisState.
# - Outputs the normalized intensity data into a new sheet titled 'Normalized to S=X'.
# Configuration:
# - Set the desired reference state using the 'reference_state' variable.
# Usage:
# - Place this script in the same directory as the Excel file.
# - Run the script to generate the normalized data directly in the same Excel file.

import pandas as pd
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's folder
os.chdir(script_dir)

# Automatically find the Excel file in the current folder
for file in os.listdir():
    if ("IM_Mask-Cholesterol-MeanInt_data" in file and 
        (file.endswith(".xlsx") or file.endswith(".xls"))):
        file_path = file
        break
else:
    raise FileNotFoundError("Excel file not found in folder.")

# Load with explicit engine based on extension
if file_path.endswith(".xlsx"):
    df = pd.read_excel(file_path, sheet_name="Raw", engine="openpyxl")
elif file_path.endswith(".xls"):
    df = pd.read_excel(file_path, sheet_name="Raw", engine="xlrd")

# Set the reference NETosisState to normalize to its first appearance
reference_state = 2  # Change as needed

# Load the "Raw" sheet
df = pd.read_excel(file_path, sheet_name="Raw")

# Drop columns with empty or NaN headers
df = df.loc[:, df.columns.dropna()]
df = df.loc[:, df.columns != ""]

# Create a new DataFrame to store normalized intensities
normalized_df = pd.DataFrame()
normalized_df["Time (frames)"] = df[df.columns[0]]

# Process each cell (group of 3 columns: time, state, intensity)
for i in range(0, df.shape[1] - 2, 3):
    time_col      = df.columns[i]
    state_col     = df.columns[i + 1]
    intensity_col = df.columns[i + 2]

    states      = df[state_col]
    intensities = df[intensity_col]

    # Find the first row where state equals reference_state
    first_match = states[states == reference_state].index
    if len(first_match) > 0:
        ref_index     = first_match[0]
        ref_intensity = intensities.loc[ref_index]
    else:
        ref_intensity = None

    # Normalize or fill with None
    if ref_intensity and ref_intensity != 0:
        normalized = intensities / ref_intensity
    else:
        normalized = [None] * len(intensities)
    
    normalized_df[f"Time_{intensity_col}"]  = df[time_col].values
    normalized_df[f"State_{intensity_col}"] = df[state_col].values
    normalized_df[f"{intensity_col} (norm@S={reference_state})"] = normalized

# Write the new sheet to the same Excel file
with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    normalized_df.to_excel(writer, sheet_name=f"Normalized to S={reference_state}", index=False)

print(f"Done! Normalized sheet 'Normalized to S={reference_state}' added to: {file_path}")