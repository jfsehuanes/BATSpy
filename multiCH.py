# This script analyzes files coming from Multi-Channel recordings

import numpy as np
import matplotlib.pyplot as plt

from IPython import embed


def get_all_ch(single_filename):
    import glob
    path = '/'.join(single_filename.split('/')[:-1])
    f = single_filename.split('/')[-1]
    all_recs = f.split('_')[0][:-1] + '*_' + '_'.join(f.split('_')[1:])
    ch_list = glob.glob('/'.join([path, all_recs]))

    return np.sort(ch_list)


def plot_multiCH_spectrogram(spec_mat, time_arr, freq_arr, pk_idxs, dynamic_range=70, ret_fig_and_ax=False):

    inch_factor = 2.54
    fs = 14
    fig, ax = plt.subplots(figsize=(56. / inch_factor, 30. / inch_factor))
    im = ax.imshow(spec_mat, cmap='jet', extent=[time_arr[0], time_arr[-1], freq_arr[0], freq_arr[-1]],
                   aspect='auto', origin='lower', alpha=0.7)

    cb_ticks = np.arange(0, dynamic_range + 10, 10)

    cb = fig.colorbar(im)

    cb.set_label('dB', fontsize=fs)
    ax.set_ylabel('Frequency [Hz]', fontsize=fs)
    ax.set_xlabel('Time [sec]', fontsize=fs)

    # plot the peaks
    colors = ['purple', 'cornflowerblue', 'forestgreen', 'darkred']
    for i in np.arange(len(pk_idxs)):
        ax.plot(time_arr[pk_idxs[i]], np.ones(len(pk_idxs[i])) * 100000 + 2000*i, 'o', ms=20, color=colors[i],
                alpha=.8, mec='k', mew=3)

    pass
