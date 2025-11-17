#https://www.youtube.com/watch?v=pfjiwxhqd1M

# 1. Create time signal
'''
fs = 500;
t = 0:1/fs:5-1/fs;
freq = 1:fs/length(t):fs-1/fs;
A = [1 4];
f = [100; 110];

xn = A*sin(2*pi*f*t);
N = length(xn);
'''

import numpy as np
fs = 500
t = np.arange(0,5,1/fs)
freq = np.arange(1, fs, fs/len(t))
A = np.array([1, 4])
f = np.array([[100], [110]])

xn = np.dot(A, np.sin(2*np.pi*f*t))
N = len(xn);


# 2. Amplitude Spectrum from FFT
'''
xk = abs(fft(xn))/N;
xk = xk(1:N/2+1);
xk(2:end-1) = 2*xk(2:end-1);

figure
plot(freq(1:N/2+1), xk, 'LineWidth', 3)
grid on
title("Amplitude Using FFT)
xlabel("Frequency(Hz)")
ylabel("Amplitude")
'''
import matplotlib.pyplot as plt
xk = np.abs(np.fft.fft(xn)) / N
xk = xk[0:int(N/2)+1]
xk[1:-1] = 2 * xk[1:-1]

plt.figure(figsize = (10,6))
freq_axis = fs * np.arange(0, N/2+1) / N
plt.plot(freq_axis, xk, linewidth=3)

plt.grid(True)
plt.title("Amplitude Using FFT")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.show()


# 3. Power Spectrum from FFT
'''
xk = (1/N^2)*abs(fft(xn)).^2; % two-sided power
xk = xk(1:N/2+1); %one-sided
xk(2:end-1) = 2*xk(2:end-1); %Double values except for DC and Nyquist

figure
#plot(freq(1:N/2+1), xk, 'LineWidth', 3)
plot(freq(1:N/2+1), pow2db(xk), 'LineWidth', 3)
grid on
title("Power Using FFT")
xlable("Frequency(Hz)")
ylable("Power")
'''
xk = (1/N**2) * np.abs(np.fft.fft(xn))**2
xk = xk[0:int(N/2)+1]
xk[1:-1] = 2*xk[1:-1]
plt.figure()
#plt.plot(freq[0:int(N/2)+1], xk, linewidth=3)
xk_db = 10 * np.log10(xk)
plt.plot(freq[0:int(N/2)+1], xk_db, linewidth=3)
plt.grid(True)
plt.title("Power Using FFT")
plt.xlabel("Frequency(Hz)")
plt.ylabel("Power(dB)")
plt.show()

# 4. Periodogram
'''
periodogram(xn, rectwin(length(xn)), length(xn), fs, 'power')
'''
from scipy import signal
freqs, power_spectrum = signal.periodogram(xn, fs=fs, scaling='spectrum')
xk_db = 10 * np.log10(power_spectrum)
plt.figure()
#plt.semilogy(freqs, power_spectrum)
plt.plot(freqs, xk_db)
plt.title("Periodogram using ScyPy")
plt.xlabel("Frequency(Hz)")
plt.ylabel("Power/Frequency(V^2/Hz")
plt.grid(True)
plt.show()

