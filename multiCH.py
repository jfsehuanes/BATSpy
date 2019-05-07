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


def plot_multiCH_spectrogram(specs_matrix, time_arr, freq_arr, pk_idxs, all_ch_peak_times,
                             filepath, dyn_range=70, in_kHz=True):

    if in_kHz:
        hz_fac = 1000
    else:
        hz_fac = 1

    inch_factor = 2.54
    fs = 18
    fig, ax = plt.subplots(nrows=len(specs_matrix), figsize=(50. / inch_factor, 25. / inch_factor),
                           sharex=True, sharey=True)
    colors = ['purple', 'cornflowerblue', 'forestgreen', 'darkred']

    for i in np.arange(len(specs_matrix)):

        # Reduce the noise by subtracting the mean power of the spectrum plus 5dB
        med = np.median(specs_matrix[i]) + 5
        specs_matrix[i][specs_matrix[i] <= med] = -dyn_range

        im = ax[i].imshow(specs_matrix[i], cmap='jet',
                          extent=[time_arr[0], time_arr[-1], int(freq_arr[0])/hz_fac, int(freq_arr[-1])/hz_fac],
                          aspect='auto', origin='lower', alpha=0.7)
        ax[i].plot(time_arr[pk_idxs[i]], np.ones(len(pk_idxs[i])) * 100, 'o', ms=7, color=colors[i],
                   alpha=.8, mec='k', mew=1.5)
        ax[i].plot(all_ch_peak_times, np.ones(len(all_ch_peak_times)) * 150, 'o', ms=7, color='gray',
                   alpha=.8, mec='k', mew=1.5)
        ax[i].tick_params(labelsize=fs-1)
    # ToDo: Plot the colorbar!
    # cb_ticks = np.arange(0, dynamic_range + 10, 10)
    # cb = fig.colorbar(im)
    # cb.set_label('dB', fontsize=fs)

    ax[1].set_ylabel('Frequency [kHz]', fontsize=fs+2)
    ax[-1].set_xlabel('Time [sec]', fontsize=fs+2)
    figtitle = '/'.join(filepath.split('/')[-4:-1]) + '/' +\
               '_'.join(filepath.split('/')[-1].split('_')[1:3]).split('.')[0]
    fig.suptitle(figtitle, fontsize=fs + 2)

    pass


def get_calls_across_channels(all_ch_filenames, run_window_width=0.05, step_quotient=10, ch_jitter_th=0.005,
                              f_res=2**9, overlap=0.7, dr=70, plot_spec=False, debug_plot=False):
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
    f_res: int power of 2
        The frequency resolution for the powerspectrum. See Batspy-class for details.
    overlap: float between 0 & 1
        Overlap fraction of the FFT windows. See Batspy-class for details.
    dr: int.
        Sets the dynamic range for spectrogram. See Batspy-class for details.
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

    for rec_idx in range(len(all_ch_filenames)):
        print('\nAnalyzing Channel ' + str(rec_idx + 1) + ' of ' + str(len(all_ch_filenames)) + '...')
        bat = Batspy(all_ch_filenames[rec_idx], f_resolution=f_res, overlap_frac=overlap, dynamic_range=dr)
        bat.compute_spectrogram()
        specs.append(bat.plt_spec)
        _, p, _ = bat.detect_calls()
        pk_arrays.append(p)
        spec_time = bat.t  # time array of the spectrogram
        spec_freq = bat.f  # frequency array of the spectrogram
        recs_info = bat.file_path

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
    callChannel = []

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
                                                  for e in range(len(norm_pow))])
            channel_sequence[enu] = channel_w_highest_pow
            c_callTimes = spec_time[win_peak_idxs[channel_w_highest_pow]]
            call_times.append(c_callTimes)
            callChannel.append(np.array([channel_w_highest_pow for m in range(len(c_callTimes))]))

    # Now there are double detections for the same call when transitioning from one channel to the other.
    # For this I need to make an average of the time call for the case with double detections.

    call_times, indi = np.unique(np.hstack(call_times), return_index=True)  # Use unique to remove same call detected in several windows
    callChannel = np.hstack(callChannel)[indi]
    reps_idx = np.where(np.diff(call_times) <= ch_jitter_th)[0]  # array with repetition indices
    replacements = np.array([call_times[e] + (call_times[e + 1] - call_times[e]) / 2. for e in reps_idx])
    call_times[reps_idx] = replacements  # first replace with average
    call_times = np.delete(call_times, reps_idx + 1)  # then remove the extra call
    callChannel = np.delete(callChannel, reps_idx + 1)

    if plot_spec:  # plot a spectrogram that includes all channels!
        plot_multiCH_spectrogram(specs, spec_time, spec_freq, pk_arrays, call_times, recs_info, dyn_range=dr)

    if debug_plot:  # plot the normed powers for debugging
        fig, ax = plt.subplots()
        colors = ['purple', 'cornflowerblue', 'forestgreen', 'darkred']
        for i in np.arange(len(specs)):
            ax.plot(spec_time, norm_pow[i], color=colors[i], alpha=.8)
            ax.plot(spec_time[pk_arrays[i]], np.ones(len(pk_arrays[i])) * 1 + 1 * i, 'o', ms=7, color=colors[i],
                    alpha=.8, mec='k', mew=3)
        ax.plot(call_times, np.ones(len(call_times)) * 6, 'o', ms=7, color='gray', alpha=.8, mec='k', mew=3)

    return call_times, callChannel


def plot_call_parameter_distributions(cp_dict, showit=True):

    if type(cp_dict[list(cp_dict.keys())[0]]) == list:
        cp_dict = {e: np.array(cp_dict[e]) for e in cp_dict.keys()}

    inch_factor = 2.54
    fs = 14
    cfb = 'cornflowerblue'
    fig, ax1 = plt.subplots(figsize=(20. / inch_factor, 15. / inch_factor))
    ax2 = ax1.twinx()

    cd = (cp_dict['ce'] - cp_dict['cb']) * 1000.  # in ms
    fb = cp_dict['fb'] / 1000.  # in kHz
    pf = cp_dict['pf'] / 1000.  # in kHz
    fe = cp_dict['fe'] / 1000.  # in kHz

    vp1 = ax1.violinplot([cd], [1], showextrema=False, widths=.8)
    vp2 = ax2.violinplot([fb, pf, fe], [2, 3, 4], showextrema=False, widths=.8)

    ax1.plot([1], [np.median(cd)], '.', color='none', mec='k', mew=2, ms=15)
    ax2.plot([2, 3, 4], [np.median(e) for e in [fb, pf, fe]], '.', color='none', mec='k', mew=2, ms=15)

    for pc in vp2['bodies']:
        pc.set_facecolor('gray')
        pc.set_edgecolor('black')
        pc.set_alpha(0.8)

    for pc2 in vp1['bodies']:
        pc2.set_facecolor(cfb)
        pc2.set_edgecolor('black')
        pc2.set_alpha(0.8)

    # embed()
    # quit()

    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xticklabels(['Call Duration', 'Fmax', 'Fpk', 'Fmin'], fontsize=fs)
    ax1.set_ylabel('Duration [ms]', fontsize=fs)
    ax2.set_ylabel('Frequency [kHz]', fontsize=fs)
    ax2.set_title('n = %.i' % len(cp_dict['call_number']), fontsize=fs)

    ax1.set_yticks(np.arange(0., 3.5, 0.5))
    ax2.set_yticks(np.arange(60., 220., 20))

    for ax in [ax1, ax2]:
        ax.tick_params(labelsize=fs-3)

    # color the first y axis
    ax1.spines['left'].set_color(cfb)
    ax2.spines['left'].set_color(cfb)
    ax1.spines['left'].set_alpha(1)
    ax2.spines['left'].set_alpha(1)
    ax1.yaxis.label.set_color(cfb)
    ax1.yaxis.label.set_alpha(1)
    ax1.tick_params(axis='y', colors=cfb)

    if showit:
        plt.show()

    pass
