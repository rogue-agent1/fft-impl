#!/usr/bin/env python3
"""Fast Fourier Transform. Zero dependencies."""
import math, sys

def fft(x):
    N = len(x)
    if N <= 1: return x
    if N % 2 != 0:
        return dft(x)
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [_exp(-2j * math.pi * k / N) * odd[k] for k in range(N//2)]
    return [even[k] + T[k] for k in range(N//2)] + [even[k] - T[k] for k in range(N//2)]

def ifft(X):
    N = len(X)
    conj = [x.conjugate() for x in X]
    result = fft(conj)
    return [x.conjugate() / N for x in result]

def dft(x):
    N = len(x)
    return [sum(x[n] * _exp(-2j*math.pi*k*n/N) for n in range(N)) for k in range(N)]

def _exp(x):
    return complex(math.cos(x.imag), math.sin(x.imag)) if isinstance(x, complex) else complex(math.cos(x), math.sin(x))

def magnitude(X):
    return [abs(x) for x in X]

def phase(X):
    return [math.atan2(x.imag, x.real) for x in X]

def power_spectrum(X):
    return [abs(x)**2 for x in X]

def frequencies(N, sample_rate):
    return [k * sample_rate / N if k <= N//2 else (k-N) * sample_rate / N for k in range(N)]

def sine_wave(freq, sample_rate, duration, amplitude=1.0):
    N = int(sample_rate * duration)
    return [complex(amplitude * math.sin(2*math.pi*freq*t/sample_rate)) for t in range(N)]

if __name__ == "__main__":
    signal = sine_wave(440, 8000, 0.01)
    X = fft(signal)
    mags = magnitude(X)
    peak = mags.index(max(mags))
    freqs = frequencies(len(signal), 8000)
    print(f"Peak frequency: {abs(freqs[peak]):.0f} Hz")
