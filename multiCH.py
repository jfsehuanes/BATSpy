''' This script analyzes files coming from Multi-Channel recordings '''

import numpy as np
import matplotlib.pyplot as plt

from bats import Batspy

from IPython import embed


def get_all_ch(single_filename):
    import glob
    path = '/'.join(single_filename.split('/')[:-1])
    f = single_filename.split('/')[-1]
    all_recs = f.split('_')[0][:-1] + '*_' + '_'.join(f.split('_')[1:])
    ch_list = glob.glob('/'.join([path, all_recs]))

    return np.sort(ch_list)


def plot_multiCH_spectrogram(specs_matrix, time_arr, freq_arr, pk_idxs, all_ch_peak_times):
    inch_factor = 2.54
    fs = 14
    fig, ax = plt.subplots(nrows=len(specs_matrix), figsize=(50. / inch_factor, 25. / inch_factor),
                           sharex=True, sharey=True)
    colors = ['purple', 'cornflowerblue', 'forestgreen', 'darkred']

    for i in np.arange(len(specs_matrix)):

        im = ax[i].imshow(specs_matrix[i], cmap='jet',
                          extent=[time_arr[0], time_arr[-1], freq_arr[0], freq_arr[-1]],
                          aspect='auto', origin='lower', alpha=0.7)
        ax[i].plot(time_arr[pk_idxs[i]], np.ones(len(pk_idxs[i])) * 100000, 'o', ms=7, color=colors[i],
                   alpha=.8, mec='k', mew=1.5)
        ax[i].plot(all_ch_peak_times, np.ones(len(all_ch_peak_times)) * 150000, 'o', ms=7, color='gray',
                   alpha=.8, mec='k', mew=1.5)
        ax[i].set_yticklabels(['{:,}'.format(int(x/1000)) for x in ax[i].get_yticks().tolist()])
    # ToDo: Plot the colorbar!
    # cb_ticks = np.arange(0, dynamic_range + 10, 10)
    # cb = fig.colorbar(im)
    # cb.set_label('dB', fontsize=fs)

    ax[1].set_ylabel('Frequency [kHz]', fontsize=fs+2)
    ax[-1].set_xlabel('Time [sec]', fontsize=fs+2)

    pass


def get_calls_across_channels(all_ch_filenames, run_window_width=0.2, step_quotient=4,
                              ch_jitter_th=0.005, plot_spec=False, debug_plot=False):
    """

    Parameters
    ----------
    all_ch_filenames: array or list.
    Array with the full path and name of the files from which calls should be extracted from
    run_window_width: float.
        Width of the window (in seconds) that runs through the mean powers in search for calls
    step_quotient: int.
        Quotient specifying the fraction by which run_window_width is divided,
        thereby setting the step_size
    ch_jitter_th: float.
        The threshold below which calls are considered repetitions (i.e. multiple detections)
        in several channels
    plot_spec: bool.
        Set to True if you wish to have spectrograms with detected calls plotted.

    debug_plot: bool.
        Set to True if you wish a plot with which one can evaluate how the calls from different
        channels are being merged together.

    Returns
    -------
    call_times: 1D array.
        Returns an array with the times (in seconds) of all detected calls.

    """

    pk_arrays = []
    specs = []

    for rec_idx in np.arange(len(all_ch_filenames)):
        print('\nAnalyzing Channel ' + str(rec_idx + 1) + ' of ' + str(len(all_ch_filenames)) + '...')
        bat = Batspy(all_ch_filenames[rec_idx], f_resolution=2 ** 9, overlap_frac=.70, dynamic_range=70)
        bat.compute_spectrogram()
        specs.append(bat.plt_spec)
        _, p, _ = bat.detect_calls()
        pk_arrays.append(p)
        spec_time = bat.t  # time array of the spectrogram
        spec_freq = bat.f  # frequency array of the spectrogram

    # The problem across channels is that there is a delay in the call from one mike to the other and
    # this results in double detections which are time-shifted.
    # I therefore need to decide on which channel to stay for a fixed period of time (running-windows).
    # The channel that registers higher amplitude values at call_times is the channel to be taken into account.

    av_pow = [np.mean(e, axis=0) for e in specs]
    norm_pow = [e - np.mean(e) for e in av_pow]  # Norm, because each mike has a different gain and therefore power
    time_sr = 1. / np.diff(spec_time[:2])[0]  # time resolution of the spectrogram

    # find window_size in index
    idx_window_width = int(np.floor(time_sr * run_window_width / 1.))
    step_size = idx_window_width // step_quotient  # step_size is 0.05s for a run_window_width of 0.2s
    last_idx = len(spec_time) - idx_window_width
    steps = np.arange(0, last_idx, step_size)

    channel_sequence = np.ones(len(steps))
    call_times = []

    for enu, step in enumerate(steps):

        # first we need the valid indices for the current window (for all 4 channels, i.e shape=(n,4))
        win_peak_idxs = [pk_arrays[e][np.logical_and(pk_arrays[e] > step,
                                                     pk_arrays[e] < step + idx_window_width)]
                         for e in np.arange(len(pk_arrays))]

        if np.sum(np.concatenate(win_peak_idxs)) == 0:
            # This is the case when there are no detections within the window
            channel_sequence[enu] = np.nan
            continue

        else:
            # now we look for the power within de valid peaks for each channel.
            # Then the channel with the highest mean is chosen. This is achieved with nanargmax,
            # which output ranges from 0 - #_of_ch (i.e. the channels)
            channel_w_highest_pow = np.nanargmax([np.mean(norm_pow[e][win_peak_idxs[e]])
                                                  for e in np.arange(len(norm_pow))])
            channel_sequence[enu] = channel_w_highest_pow
            call_times.append(spec_time[win_peak_idxs[channel_w_highest_pow]])

    # Now there are double detections for the same call when transitioning from one channel to the other.
    # For this I need to make an average of the time call for the case with double detections.

    call_times = np.unique(np.hstack(call_times))  # Use unique to remove same call detected in several windows
    reps_idx = np.where(np.diff(call_times) <= ch_jitter_th)[0]  # array with repetition indices
    replacements = np.array([call_times[e] + (call_times[e + 1] - call_times[e]) / 2. for e in reps_idx])
    call_times[reps_idx] = replacements  # first replace with average
    call_times = np.delete(call_times, reps_idx + 1)  # then remove the extra call

    if plot_spec:  # plot a spectrogram that includes all channels!
        plot_multiCH_spectrogram(specs, spec_time, spec_freq, pk_arrays, call_times)

    if debug_plot:  # plot the normed powers for debugging
        colors = ['purple', 'cornflowerblue', 'forestgreen', 'darkred']
        for i in np.arange(len(specs)):
            plt.plot(spec_time, norm_pow[i], color=colors[i], alpha=.8)
            plt.plot(spec_time[pk_arrays[i]], np.ones(len(pk_arrays[i])) * 1 + 1 * i, 'o', ms=7, color=colors[i],
                     alpha=.8, mec='k', mew=3)
        plt.plot(call_times, np.ones(len(call_times)) * 6, 'o', ms=7, color='gray', alpha=.8, mec='k', mew=3)

    return call_times
