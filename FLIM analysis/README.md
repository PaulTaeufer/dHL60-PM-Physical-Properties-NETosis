# IRF Calculation (irf-calc.py)

This README describes the purpose, usage, inputs, outputs and dependencies for the script `irf-calc.py` located in this project. The script performs instrument response function (IRF) related calculations used in fluorescence lifetime imaging (FLIIM) data processing.

Summary
- Purpose: compute, normalize, and export instrument response functions (IRFs) and related metadata used for deconvolution of fluorescence decay traces.
- Typical tasks: load raw IRF measurement(s), resample or interpolate to desired time bins, normalize area or peak, optionally fit a parametric model (e.g., Gaussian), and save processed IRF for use by lifetime-fitting routines.

Usage
- From the command line: python irf-calc.py [options]
- Common options (typical; script-specific flags may differ):
	-i / --input PATH    Input file or folder with raw IRF measurement(s)
	-o / --output PATH   Output file to save processed IRF (e.g., .npy, .csv)
	-b / --bins N        Number of time bins or desired time resolution
	-r / --resample      Resample/interpolate IRF to match target time grid
	-n / --normalize     Normalize IRF area or peak
	-f / --fit MODEL     Fit a model (gaussian, lorentzian, etc.) and save parameters

Inputs
- Raw IRF data: common formats include text/csv, binary numpy (.npy), or image formats containing time-resolved counts.
- Time axis: either implicit (index-based) or an explicit time vector accompanying counts.

Outputs
- Processed IRF: resampled and normalized IRF saved to disk for downstream deconvolution and lifetime fitting.
- Optional fit parameters: JSON or text file containing fitted model parameters and goodness-of-fit metrics.

Integration notes
- Ensure the IRF timebase (time per bin) matches the decay data timebase used by your lifetime fitting routines.
- When normalizing the IRF, document whether area or peak normalization was used — this affects amplitude and fitted lifetimes.

Examples
- Resample and normalize an IRF and save as numpy array:
	python irf-calc.py -i raw_irf.csv -o irf_processed.npy -b 1024 -r -n