TEST TO SEE GP Value Analysis Pipeline for Fluorescence Microscopy (CZI files)
===================================================================
Calculates Generalized Polarization (GP) values for membrane-stained cells.

GP = (I_440 - I_490_mean) / (I_440 + I_490_mean)
where I_490_mean = mean of two 490nm measurements (bleaching correction)

Channel layout (auto-detected, adjust CH_* constants if needed):
  C0: Brightfield (transmitted)
  C1: 490nm measurement A
  C2: 440nm (also used for segmentation)
  C3: 490nm measurement B
  C4: 650nm Cy5 (SpyDNA, not used in GP)

Usage:
  Place this script in the folder that contains your CZI images and run:
      python gp_analysis.py

Outputs (saved in ./results/ subfolder):
  - results/gp_results.xlsx                  : all cells, all images, one sheet
  - THRESHOLD_FACTOR      : fraction of Otsu threshold (1.0 = bright compact cells only,
                             0.3 = also catches dim flat adhered cells — default)
  - MIN_CELL_AREA / MAX_CELL_AREA : size filter in px²
  - results/<image>/<image>_envelope.tif     : labeled membrane-ring mask
  - results/<image>/<image>_overview.png     : annotated overview image
