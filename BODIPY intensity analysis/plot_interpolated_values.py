import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# This script loads experimental data from an Excel file, processes it, and visualizes
# fluorescence intensity changes across different NETosis states.

# Steps performed:
# 1. Automatically locates and loads the Excel file in the script’s folder.
# 2. Cleans and converts measurement columns to numeric values.
# 3. Creates a continuous x-axis representing progression within and across NETosis states.
# 4. Identifies turning points (state transitions) for labeling.
# 5. Plots individual cell traces and their average curve, aligned to biological events.
# 6. Formats the plot with custom axis labels, event markers, and a clear layout.

# Result: A figure showing the dynamics of interpolated BODIPY mean intensity relative
# to key NETosis events, allowing comparison between single-cell data and the population average.

# --- Load the Excel sheet ---
# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's folder
os.chdir(script_dir)

# Automatically find the Excel file in the current folder
for file in os.listdir():
    if file.endswith(".xlsx") and "Mask-Cholesterol-MeanInt_data" in file:
        file_path = file
        break
else:
    raise FileNotFoundError("Excel file not found in folder.")
df = pd.read_excel(file_path, sheet_name="Interpolated", decimal=",")

# alle Messspalten numerisch machen (NaNs erlaubt)
for c in df.columns[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ===== x-Achse: kontinuierlicher Fortschritt innerhalb jedes States =====
state = df["NETosisState"].astype(int)
within_idx = state.groupby(state).cumcount()                 # 0,1,2,... innerhalb des States
counts = state.map(state.value_counts())                     # Länge je State
frac = np.where(counts > 1, within_idx / (counts - 1), 0.0)  # 0..1 innerhalb des States
x = state.values + frac                                      # z.B. 0..1..2..3 kontinuierlich

# ===== Wendepunkte (erster Frame jedes neuen States) + Endlabel =====
first_rows = df.groupby(state, sort=True).head(1)
turn_states = first_rows["NETosisState"].astype(int).tolist()
turn_positions = (first_rows["NETosisState"].values + 0.0).tolist()  # erster Punkt im State liegt bei frac=0
turn_positions.append(x.max())                                       # ganz rechts: Ende Film

# State-Namen (anpassen falls andere Bezeichnungen)
state_name = {0: "0", 1: "MV shedding", 2: "PM rupture / premeabilization"}  # 4 ggf. anpassen/ergänzen
tick_labels = [state_name.get(s, f"State {s}") for s in turn_states] + ["End of movie"]

# ===== Plot =====
fig, ax = plt.subplots(figsize=(11,5))

# jede Zelle
value_cols = df.columns[1:]
for col in value_cols:
    ax.plot(x, df[col].values, lw=2, label=col)

# Durchschnittskurve (schwarz, dicker)
avg = df[value_cols].mean(axis=1, skipna=True)
ax.plot(x, avg, lw=3, color="k", label="Average")

# Achsen & Look (ähnlich zum Beispiel)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0, 6)  # bei Bedarf anpassen
ax.set_ylabel("Interpolated Mean Int relative to MV shedding")
ax.set_title("Interpolated BODIPY Mean Int by Events - IM (20250317/27)")

# nur an Wendepunkten & am Ende beschriften
ax.set_xticks(turn_positions, tick_labels)

# Spines/Legende
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

plt.tight_layout()
plt.show()