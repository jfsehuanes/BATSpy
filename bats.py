import numpy as np
import matplotlib.pyplot as plt

from dataloader import load_data
from powerspectrum import spectrogram, decibel

from IPython import embed


if __name__ == '__main__':

    # Get the data
    recording = '../../data/pc-tape_recordings/' \
                'macaregua__february_2018/natalus_outside_cave/natalusTumidirostris0045.wav'

    d, sr, u = load_data(recording)
    sr *= 10.  # This fixes PC-Tape's bug that writes 1/10 of the samplingrate in the header of the .wav file

    f_res = 1000.  # in Hz
    spec, f, t = spectrogram(d.squeeze(), sr, f_res, overlap_frac=0.8)
    # in imshow, parameter extent sets the canvas edges!
    plt.imshow(decibel(spec)[::-1], extent=[t[0], t[-1], f[0], f[-1]], aspect='auto', alpha=0.7, cmap='jet')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    # embed()
    # quit()