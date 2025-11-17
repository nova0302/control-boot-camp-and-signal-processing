import numpy as np

class MovingAverageFilter:
    def __init__(self, n=100):
        self.n = n
        self.xbuf = None

    def update(self, x):
        if self.xbuf is None:
            self.xbuf = np.full((self.n, 1), x)

        # Shift and insert new value
        self.xbuf[0:self.n-1] = self.xbuf[1:self.n]
        self.xbuf[self.n-1] = x

        # Calculate the average
        avg = np.sum(self.xbuf) / self.n
        return avg

# Usage of the class:
if __name__ == '__main__':
    filter_instance = MovingAverageFilter(n=4)
    data_stream = [1, 2, 3, 4, 5, 100, 6, 7, 8, 9]
    arr = np.array(data_stream)
    arr_mean = arr.mean()
    print(f'mean = {arr_mean}')
    for value in data_stream:
        average_output = filter_instance.update(value)
        print(f'average_output = {average_output}')
        # ...
