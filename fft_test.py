'''
fs = 40;
T = 1/fs;
N = 40;
t = (0:N-1)*T';

Xn = sing(2*pi*3*t);
plot(t, Xn, '.-', 'MarkerSize', 2);
title("Signal")
xlabel("t")
ylabel("Xn(t)")

Y = fft(Xn)
figure
stem(abs(Y(1:(N/2)+1)))
title("Absolute Value of the FFT of Xn")


'''

import numpy as np
import matplotlib.pyplot as plt

fs = 40
T = 1/fs
N = 40

t = np.arange(N)*T
t

Xn = np.sin(2*np.pi*3*t)

'''
plt.figure(figsize=(10,6))
plt.plot(t, Xn, '.-', markersize=10)
plt.title("Signal")
plt.xlabel("t")
plt.ylabel("Xn(t)")
plt.grid(True)
plt.show()
'''

Y = np.fft.fft(Xn)
k = int(N/2)+1
freq = np.arange(k)*fs/N

#plt.stem(np.arange(N), np.abs(Y))
plt.stem(freq, np.abs(Y[:k]))
plt.title("Absolute Value of the FFT of Xn (Frequency Domain)")
plt.xlabel("Frequency bin (k)")
plt.ylabel("|Y[k]|")
plt.grid(True)
plt.show()

