import numpy as np
import audioio

from IPython import embed

if __name__ == '__main__':
    f0 = 180000.  # Start frequency in Hz
    fe = 90000.  # End frequency in Hz
    t0 = 0.  # Start time in s
    te = 0.2  # End time in s
    samp_freq = 600. * 1000.  # in Hz
    time = np.arange(t0, te, 1./samp_freq)

    # Define the slope of the phase function
    m = (fe - f0) / (te - t0)

    # Phase integral function
    phase = m / 2. * time ** 2 + f0 * time

    # Signal
    s = np.sin(2. * np.pi * phase)

    # Save the signal
    out_file = 'test_result/stim.wav'
    audioio.write_audio(out_file, s, samp_freq, 'wav')

    # ToDo: Make a signal 5s long containing multiple 2ms sweeps
