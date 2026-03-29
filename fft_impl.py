#!/usr/bin/env python3
"""fft_impl - Cooley-Tukey FFT and inverse FFT implementation."""
import sys, math

def fft(x):
    n = len(x)
    if n <= 1: return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [math.e**(-2j*math.pi*k/n) * odd[k] for k in range(n//2)]
    return [even[k] + T[k] for k in range(n//2)] + [even[k] - T[k] for k in range(n//2)]

def ifft(X):
    n = len(X)
    conj = [x.conjugate() for x in X]
    result = fft(conj)
    return [x.conjugate() / n for x in result]

def magnitude(X):
    return [abs(x) for x in X]

def test():
    signal = [1, 0, 1, 0, 1, 0, 1, 0]
    F = fft(signal)
    assert len(F) == 8
    assert abs(F[0] - 4) < 1e-9  # DC component
    recovered = ifft(F)
    for i in range(len(signal)):
        assert abs(recovered[i].real - signal[i]) < 1e-9
    sine = [math.sin(2*math.pi*k/8) for k in range(8)]
    F2 = fft(sine)
    mags = magnitude(F2)
    peak = max(range(len(mags)), key=lambda i: mags[i])
    assert peak == 1 or peak == 7  # frequency 1
    print("fft_impl: all tests passed")

if __name__ == "__main__":
    test() if "--test" in sys.argv else print("Usage: fft_impl.py --test")
