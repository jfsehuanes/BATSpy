import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from dataloader import load_data
from powerspectrum import spectrogram, decibel

from IPython import embed


def correct_spectogram_ratio(spec_data, nfft, samplerate, start_time, end_time, noice_cancel=False):

    if noice_cancel:
        mean_channel_power = np.mean(spec_data, axis=0)
        a = spec_data - mean_channel_power
    else:
        a = spec_data

    comp_max_freq = 240000  # I guess this is the maximal frequency we want to see in the plot [in Hz]
    comp_min_freq = 0  # I guess this is the maximal frequency we want to see in the plot [in Hz]
    spectra = a[0]
    spec_freqs = a[1]
    spec_times = a[2]

    start_idx = 0
    tmp_times = spec_times - ((nfft / samplerate) / 2) + (start_idx / samplerate)
    # if nffts_per_psd == 1:
    #     tmp_times = spec_times - ((nfft / samplerate) / 2) + (start_idx / samplerate)
    # else:
    #     tmp_times = spec_times[:-(nffts_per_psd - 1)] - ((nfft / samplerate) / 2) + (start_idx / samplerate)

    # etxtract reduced spectrum for plot
    plot_freqs = spec_freqs[spec_freqs < comp_max_freq]
    plot_spectra = spectra[spec_freqs < comp_max_freq]

    fig_xspan = 20.
    fig_yspan = 12.
    fig_dpi = 80.
    no_x = fig_xspan * fig_dpi
    no_y = fig_yspan * fig_dpi

    min_x = start_time
    max_x = end_time

    min_y = comp_min_freq
    max_y = comp_max_freq

    x_borders = np.linspace(min_x, max_x, no_x * 2)
    y_borders = np.linspace(min_y, max_y, no_y * 2)
    # checked_xy_borders = False

    tmp_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))

    recreate_matrix = False
    if (tmp_times[1] - tmp_times[0]) > (x_borders[1] - x_borders[0]):
        x_borders = np.linspace(min_x, max_x, (max_x - min_x) // (tmp_times[1] - tmp_times[0]) + 1)
        recreate_matrix = True
    if (spec_freqs[1] - spec_freqs[0]) > (y_borders[1] - y_borders[0]):
        recreate_matrix = True
        y_borders = np.linspace(min_y, max_y, (max_y - min_y) // (spec_freqs[1] - spec_freqs[0]) + 1)
    if recreate_matrix:
        tmp_spectra = np.zeros((len(y_borders) - 1, len(x_borders) - 1))

    for i in range(len(y_borders) - 1):
        for j in range(len(x_borders) - 1):
            if x_borders[j] > tmp_times[-1]:
                break
            if x_borders[j + 1] < tmp_times[0]:
                continue

            t_mask = np.arange(len(tmp_times))[(tmp_times >= x_borders[j]) & (tmp_times < x_borders[j + 1])]
            f_mask = np.arange(len(plot_spectra))[(plot_freqs >= y_borders[i]) & (plot_freqs < y_borders[i + 1])]

            if len(t_mask) == 0 or len(f_mask) == 0:
                continue

            tmp_spectra[i, j] = np.max(plot_spectra[f_mask[:, None], t_mask])
    return tmp_spectra


if __name__ == '__main__':

    # Get the data
    recording = '../../data/pc-tape_recordings/' \
                'macaregua__february_2018/natalus_outside_cave/natalusTumidirostris0012.wav'

    d, sr, u = load_data(recording)
    sr *= 10.  # This fixes PC-Tape's bug that writes 1/10 of the samplingrate in the header of the .wav file

    f_res = 1000.  # in Hz
    spec, f, t = spectrogram(d.squeeze(), sr, f_res, overlap_frac=0.9)

    # remove mean noise
    mean_channel_power = np.mean(spec)
    spec -= mean_channel_power

    # in imshow, parameter extent sets the canvas edges!
    # ToDo: Reset colorbar scale! I don't get the blues, because I have no negative values
    plt.imshow(decibel(spec), cmap='jet', extent=[t[0], t[-1], f[0], f[-1]], aspect='auto', origin='lower',
               alpha=0.7)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
