import numpy as np
import matplotlib.pyplot as plt

def load_and_plot_single_column(filename):
    """
    Loads data from a single-column text file and plots it.
    The X-axis will be the index (0, 1, 2, ... N-1).
    """
    try:
        # Load just one array of Y values
        y_values = np.loadtxt(filename)
        
        # Create an X-axis array based on the length of the Y data
        x_values = np.arange(len(y_values))
        
        print(f"Loaded {len(y_values)} data points.")

        # --- Plotting the data ---
        plt.figure(figsize=(8, 5))
        # Use 'o-' to show both the line and the actual data points
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')
        
        # Add titles and labels
        plt.title(f'Single Column Data from {filename}')
        plt.xlabel('Data Index / Time Step')
        plt.ylabel('Value (Y Axis)')
        plt.grid(True)
        
        # Display the plot
        plt.show()

    except IOError as e:
        print(f"Error: Could not read file {filename}")
        print(f"Details: {e}")
    except ValueError as e:
        print(f"Error: Data format in {filename} is incorrect. Ensure all entries are numbers.")
        print(f"Details: {e}")

# --- Run the function with your filename ---
if __name__ == "__main__":
    load_and_plot_single_column('sonarAlt.txt')


