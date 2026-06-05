# Intensity Normalization + Interpolation Tool
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Automatically find the Excel file in the current folder
for file in os.listdir():
    if file.endswith(".xlsx") and "IM_Mask-Cholesterol-MeanInt_data" in file:
        file_path = file
        break
else:
    raise FileNotFoundError("Excel file not found in folder.")

# --- CONFIGURATION ---
reference_state    = 2   # Normalize to first appearance of this state
interpolation_points = 100  # Number of interpolated points per state
do_interpolation   = True   # Set to False to skip interpolation sheet
# ----------------------

# Load the "Raw" sheet
df = pd.read_excel(file_path, sheet_name="Raw", engine="openpyxl")

# Drop columns with empty or NaN headers
df = df.loc[:, df.columns.dropna()]
df = df.loc[:, df.columns != ""]

# ── NORMALIZATION ──────────────────────────────────────────────────────────────

normalized_df = pd.DataFrame()

for i in range(0, df.shape[1] - 2, 3):
    time_col      = df.columns[i]
    state_col     = df.columns[i + 1]
    intensity_col = df.columns[i + 2]

    states      = df[state_col]
    intensities = df[intensity_col]

    # Find first row where state equals reference_state
    first_match = states[states == reference_state].index
    if len(first_match) > 0:
        ref_index     = first_match[0]
        ref_intensity = intensities.loc[ref_index]
    else:
        ref_intensity = None

    if ref_intensity and ref_intensity != 0:
        normalized = intensities / ref_intensity
    else:
        normalized = pd.Series([None] * len(intensities))

    cell_name = intensity_col

    normalized_df[f"Time_{cell_name}"]  = df[time_col].values
    normalized_df[f"State_{cell_name}"] = df[state_col].values
    normalized_df[f"{cell_name} (norm@S={reference_state})"] = normalized.values

# ── INTERPOLATION ──────────────────────────────────────────────────────────────

if do_interpolation:
    interp_df_total = pd.DataFrame()

    for i in range(0, df.shape[1] - 2, 3):
        state_col     = df.columns[i + 1]
        intensity_col = df.columns[i + 2]
        cell_name     = intensity_col

        norm_col  = f"{cell_name} (norm@S={reference_state})"
        state_col_norm = f"State_{cell_name}"

        # Skip if normalization failed for this cell
        if norm_col not in normalized_df.columns:
            continue
        if normalized_df[norm_col].isna().all():
            continue

        cell_interp_df = pd.DataFrame()

        states_present = normalized_df[state_col_norm].dropna().unique()

        for k in sorted(states_present):
            mask      = normalized_df[state_col_norm] == k
            norm_vals = normalized_df.loc[mask, norm_col].dropna().reset_index(drop=True)

            if len(norm_vals) >= 2:
                interp_func = interp1d(
                    range(len(norm_vals)),
                    norm_vals,
                    kind='linear'
                )
                new_x = np.linspace(0, len(norm_vals) - 1, interpolation_points)
                new_y = interp_func(new_x)
            elif len(norm_vals) == 1:
                # Only one point — fill all slots with that value
                new_y = np.full(interpolation_points, norm_vals.iloc[0])
            else:
                new_y = np.full(interpolation_points, np.nan)

            chunk = pd.DataFrame({
                f"State_{cell_name}":                       [int(k)] * interpolation_points,
                f"InterpIndex_{cell_name}":                 list(range(interpolation_points)),
                f"{cell_name} (norm@S={reference_state})": new_y
            })
            cell_interp_df = pd.concat([cell_interp_df, chunk], ignore_index=True)

        # Merge this cell's interpolated data side-by-side
        if interp_df_total.empty:
            interp_df_total = cell_interp_df
        else:
            interp_df_total = pd.concat(
                [interp_df_total.reset_index(drop=True),
                 cell_interp_df.reset_index(drop=True)],
                axis=1
            )

# ── MEAN TRAJECTORY ───────────────────────────────────────────────────────────

if do_interpolation and not interp_df_total.empty:
    norm_suffix = f"(norm@S={reference_state})"

    # Collect all value columns grouped by state
    states_all = sorted(set(
        int(v)
        for col in interp_df_total.columns
        if col.startswith("State_")
        for v in interp_df_total[col].dropna().unique()
    ))

    mean_df = pd.DataFrame()

    for k in states_all:
        # Find all normalized value columns for this state
        # For each cell, grab rows where its State column == k
        state_values = []
        for col in interp_df_total.columns:
            if not col.startswith("State_"):
                continue
            cell_id   = col[len("State_"):]
            val_col   = f"{cell_id} {norm_suffix}"
            if val_col not in interp_df_total.columns:
                continue
            mask = interp_df_total[col] == k
            vals = interp_df_total.loc[mask, val_col].reset_index(drop=True)
            if len(vals) == interpolation_points:
                state_values.append(vals)

        if state_values:
            stacked   = pd.concat(state_values, axis=1)
            mean_vals = stacked.mean(axis=1)
            sem_vals  = stacked.sem(axis=1)
        else:
            mean_vals = pd.Series([np.nan] * interpolation_points)
            sem_vals  = pd.Series([np.nan] * interpolation_points)

        chunk = pd.DataFrame({
            "State":              [int(k)] * interpolation_points,
            "InterpIndex":        list(range(interpolation_points)),
            "Mean (norm)":        mean_vals.values,
            "SEM (norm)":         sem_vals.values,
        })
        mean_df = pd.concat([mean_df, chunk], ignore_index=True)

    # Separate mean from per-cell data by two empty columns
    spacer = pd.DataFrame({" ": [None] * len(interp_df_total),
                           "  ": [None] * len(interp_df_total)})
    # Pad mean_df to same length as interp_df_total if needed
    if len(mean_df) < len(interp_df_total):
        pad = pd.DataFrame(
            np.nan,
            index=range(len(interp_df_total) - len(mean_df)),
            columns=mean_df.columns
        )
        mean_df = pd.concat([mean_df, pad], ignore_index=True)
    elif len(mean_df) > len(interp_df_total):
        spacer = pd.DataFrame({" ": [None] * len(mean_df),
                               "  ": [None] * len(mean_df)})
        interp_df_total = pd.concat([
            interp_df_total,
            pd.DataFrame(np.nan, index=range(len(mean_df) - len(interp_df_total)),
                         columns=interp_df_total.columns)
        ], ignore_index=True)

    interp_df_total = pd.concat(
        [interp_df_total.reset_index(drop=True),
         spacer.reset_index(drop=True),
         mean_df.reset_index(drop=True)],
        axis=1
    )

# ── WRITE SHEETS ───────────────────────────────────────────────────────────────

with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    normalized_df.to_excel(
        writer,
        sheet_name=f"Normalized to S={reference_state}",
        index=False
    )
    if do_interpolation:
        interp_df_total.to_excel(
            writer,
            sheet_name=f"Interpolated (S={reference_state})",
            index=False
        )

print(f"Done! Sheets written to: {file_path}")