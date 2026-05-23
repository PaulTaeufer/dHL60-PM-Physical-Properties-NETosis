import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import CubicSpline


# Example usage
csv_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "17/cell4/irf.csv")  #CSV file path
png_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "17/cell4/irf.png")

try:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Check if the necessary columns are present
    if 't' not in df.columns or 'irf_ch1' not in df.columns:
        raise ValueError("The CSV file does not contain the required columns 't' and 'irf_ch1'.")

    # Separate the columns
    t_column = df['t']
    irf_ch1_column = df['irf_ch1']

except Exception as e:
    print(f"An error occurred: {e}")



#spline maker
cs = CubicSpline(np.array(t_column), np.array(irf_ch1_column))
# Generate new x values for interpolation
x_new = np.linspace(t_column.min(), t_column.max(), 10000)
y_new = cs(x_new)

#find IRF shift when Chi2 is 1
closest_index = np.argmin(np.abs(y_new - 1))
IRF_SHIFT = x_new[closest_index]

plt.scatter(t_column,irf_ch1_column)
plt.plot(x_new,y_new,color='red')
plt.axvline(x=IRF_SHIFT, color='black', linestyle=':', linewidth=2, label='Vertical Line')
plt.axhline(y=y_new[closest_index], color='black', linestyle=':', linewidth=2, label='Horizontal Line')
plt.title('IRF shift calculator: '+str(IRF_SHIFT)+' ps')
plt.xlabel('IRF [ps]')
plt.ylabel('Chi2 value')
plt.savefig(png_file_path)
print(IRF_SHIFT)
print('done')