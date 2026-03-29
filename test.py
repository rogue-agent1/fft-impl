from fft_impl import fft, ifft, magnitude, sine_wave, frequencies
import math
signal = sine_wave(100, 1000, 0.064)
X = fft(signal)
mags = magnitude(X)
freqs = frequencies(len(signal), 1000)
peak_idx = mags.index(max(mags[1:]))
assert abs(abs(freqs[peak_idx]) - 100) < 20, f"Peak at {freqs[peak_idx]}"
recovered = ifft(X)
assert abs(recovered[0].real - signal[0].real) < 0.01
print("FFT tests passed")