import scipy.io
import numpy as np
import os

def convert_mat_to_txt(mat_file_path, output_txt_path):
    # Load the .mat file (it acts like a Python dictionary)
    mat_data = scipy.io.loadmat(mat_file_path)
    print(mat_data['sonarAlt'])
    
    # Print keys to find your variable names
    print(f"Variables found in {mat_file_path}: {mat_data.keys()}")
    
    # *** IMPORTANT: Replace 'your_variable_name' with the actual key/variable name from the output above ***
    variable_name = 'sonarAlt' 
    
    if variable_name in mat_data:
        data_array = mat_data[variable_name]
        
        # Save the numpy array to a plain text file (space-delimited)
        np.savetxt(output_txt_path, data_array, delimiter='\n') # Use comma delimiter for CSV format
        print(f"Successfully converted '{variable_name}' to '{output_txt_path}'")
    else:
        print(f"Error: Variable '{variable_name}' not found in the .mat file.")

# --- Example Usage ---
# Ensure you change these paths to your actual file locations
input_file = 'SonarAlt.mat' 
output_file = 'SonarAlt.txt'

convert_mat_to_txt(input_file, output_file)

