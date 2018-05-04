import numpy as np
from scipy.io.wavfile import write

from IPython import embed

if __name__ == '__main__':
    f0 = 1000.  # Start frequency in Hz
    fe = 200. * 1000.  # End frequency in Hz
    t0 = 0.  # Start time in s
    te = 1  # End time in s
    samp_freq = 500. * 1000.  # in Hz
    time = np.arange(t0, te, 1./samp_freq)

    # Define the slope of the phase function
    m = (fe - f0) / (te - t0)

    # Phase integral function
    phase = m / 2. * time ** 2 + f0 * time

    # Signal
    s = np.sin(2. * np.pi * phase)

    # Save the signal
    out_file = 'test_result/stim.wav'
    write(out_file, int(samp_freq), s)

    # ToDo: Make a signal 5s long containing multiple 2ms sweeps
