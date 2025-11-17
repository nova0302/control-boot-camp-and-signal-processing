import numpy as np

class AvgFilter:
    """
    Implements a persistent average filter similar to the provided MATLAB code.
    This uses an IIR filter with an adaptive alpha: alpha = (k-1)/k
    """
    def __init__(self):
        # Initialize persistent variables within the instance
        self.k = 1
        self.prevAvg = 0
        self.avg = 0

    def update(self, x):
        """Processes a new input value x and returns the current average."""

        # Calculate the adaptive alpha value
        # Note: Need to handle division by zero if k somehow resets to 0, 
        # but with the current logic k always increments starting from 1.
        alpha = (self.k - 1) / self.k
        
        # Calculate the current average
        self.avg = alpha * self.prevAvg + (1 - alpha) * x
        
        # Update state for the next call
        self.prevAvg = self.avg
        self.k += 1

        return self.avg

    def GetVolt(self):
        w = 0 + 4 * np.random.randn()
        z = 14.4 + w
        return np.round(z)
class MovAvgFilter:
    def __init__(self):
        self.n = 1
        self.prevAvg = 0
        self.avg = 0;
    def update(self, x):
        self.avg = self.prevAvg + (x - x

# --- Example Usage ---
if __name__ == '__main__':
    # Create an instance of the filter
    filter_instance = AvgFilter()

    # Test data stream
    data = [10, 12, 14, 16, 18, 20]
    
    print(f"{'Input (x)':<10} | {'Output (avg)':<12} | {'k':<5} | {'alpha':<8}")
    print("-" * 50)

    for value in data:
        value = filter_instance.GetVolt()
        average = filter_instance.update(value)
        alpha = (filter_instance.k - 2) / (filter_instance.k - 1) # Print the alpha used in the *last* step
        print(f"{value:<10} | {average:<12.4f} | {filter_instance.k - 1:<5} | {alpha:<8.4f}")
