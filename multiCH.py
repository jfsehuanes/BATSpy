''' This script analyzes files coming from Multi-Channel recordings '''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from bats import Batspy
from thunderfish.powerspectrum import decibel

from IPython import embed


def get_all_ch(single_filename):
    import glob
    path = '/'.join(single_filename.split('/')[:-1])
    f = single_filename.split('/')[-1]
    all_recs = f.split('_')[0][:-1] + '*_' + '_'.join(f.split('_')[1:])
    ch_list = glob.glob('/'.join([path, all_recs]))

    return np.sort(ch_list)


def load_all_channels(single_ch_name, f_res=2 ** 9, overlap=0.7, dr=50, ret_call_pks=False, npy_file=None):
    """

    Parameters
    ----------
    single_ch_name: str with the file-path and filename of the file to be analyzed
    f_res: int power of 2
        The frequency resolution for the powerspectrum. See Batspy-class for details.
    overlap: float between 0 & 1
        Overlap fraction of the FFT windows. See Batspy-class for details.
    dr: int.
        Sets the dynamic range for spectrogram. See Batspy-class for details.
    ret_call_pks

    Returns
    -------

    """
    if npy_file is not None:
        # load specs
        # load times
        # load freqs
        # return them all
        pass
    else:
        all_ch_filenames = get_all_ch(single_ch_name)

        pk_arrays = []
        specs = []

        for rec_idx in range(len(all_ch_filenames)):
            print('\nLoading Channel ' + str(rec_idx + 1) + ' of ' + str(len(all_ch_filenames)) + '...')
            bat = Batspy(all_ch_filenames[rec_idx], f_resolution=f_res, overlap_frac=overlap, dynamic_range=dr)
            bat.compute_spectrogram()
            specs.append(bat.spec_mat)
            _, p = bat.detect_calls()
            pk_arrays.append(p)
            spec_time = bat.spec_params[0]  # time array of the spectrogram
            spec_freq = bat.spec_params[1]  # frequency array of the spectrogram
            recs_info = bat.file_path

        spec_params = np.array([spec_time,
                               spec_freq])

        if ret_call_pks:
            return specs, spec_params, pk_arrays
        else:
            return specs, spec_params


def plot_multiCH_spectrogram(specs_matrix, spec_params, filepath, dyn_range=50, in_kHz=True, adjust_to_max_db=True,
                             input_fig=None, ret_fig_chAxs_and_callAx=False):

    # ToDo: Segregate between plotting the spectrogram only and plotting the calls!

    if in_kHz:
        hz_fac = 1000
    else:
        hz_fac = 1

    inch_factor = 2.54
    fs = 18

    if input_fig is None:
        fig = plt.figure(constrained_layout=True, figsize=(60. / inch_factor, 30. / inch_factor))
        figtitle = '/'.join(filepath.split('/')[-4:-1]) + '/' + \
                   '_'.join(filepath.split('/')[-1].split('_')[1:3]).split('.')[0]
        fig.suptitle(figtitle, fontsize=fs + 2)
    else:
        fig = input_fig

    gs = gridspec.GridSpec(len(specs_matrix)+1, 2, height_ratios=(1.9, 1.9, 1.9, 1.9, 0.4), width_ratios=(9.9, .1))
    ch1 = fig.add_subplot(gs[0, :-1])
    ch2 = fig.add_subplot(gs[1, :-1])
    ch3 = fig.add_subplot(gs[2, :-1])
    ch4 = fig.add_subplot(gs[3, :-1])
    calls_ax = fig.add_subplot(gs[4:, :-1])
    cbar_ax = fig.add_subplot(gs[0:, -1])

    gs.update(wspace=0.05, hspace=0.2)

    ch_axs = [ch1, ch2, ch3, ch4]

    for i in np.arange(len(specs_matrix)):

        mat = specs_matrix[i]

        if adjust_to_max_db:
            # set dynamic range
            dec_spec = decibel(mat)
            ampl_max = np.nanmax(
                dec_spec)  # define maximum; use nanmax, because decibel function may contain NaN values
            dec_spec -= ampl_max + 1e-20  # subtract maximum so that the maximum value is set to lim x--> -0

            # Fix NaNs issue
            if True in np.isnan(dec_spec):
                dec_spec[np.isnan(dec_spec)] = - dyn_range

        im = ch_axs[i].imshow(dec_spec, cmap='jet', extent=[spec_params[0][0], spec_params[0][-1],
                                                            int(spec_params[1][0])/hz_fac,
                                                            int(spec_params[1][-1])/hz_fac],
                              aspect='auto', interpolation='hanning', origin='lower', alpha=0.7, vmin=-dyn_range,
                              vmax=0., rasterized=True)

        # Remove time ticks of the spectrogram
        ch_axs[i].xaxis.set_major_locator(plt.NullLocator())

    # Plot the colorbar
    cb = fig.colorbar(im, cax=cbar_ax)

    # Share the axes of spectrograms and the all calls plot
    ch1.get_shared_y_axes().join(ch1, ch2, ch3, ch4)
    calls_ax.get_shared_x_axes().join(ch1, ch2, ch3, ch4, calls_ax)
    calls_ax.set_xlim(spec_params[0][0], spec_params[0][-1])

    # Remove ticks
    calls_ax.yaxis.set_major_locator(plt.NullLocator())

    # Labels
    cb.set_label('dB', fontsize=fs + 4)
    ch_axs[2].set_ylabel('Frequency [kHz]', fontsize=fs+4)
    calls_ax.set_xlabel('Time [sec]', fontsize=fs+4)

    ch_axs.extend([calls_ax, cbar_ax])
    for c_ax in ch_axs:
        c_ax.tick_params(labelsize=fs)

    if ret_fig_chAxs_and_callAx:
        return fig, ch_axs, calls_ax
    else:
        pass


def plot_calls_in_spectrogram(spec_fig, ch_axs, pk_idxs, time_arr, calls_ax, all_ch_peak_times, showit=False):

    # Define colors of the call detection for each channel
    colors = ['purple', 'cornflowerblue', 'forestgreen', 'darkred']

    for i in range(len(pk_idxs)):
        ch_axs[i].plot(time_arr[pk_idxs[i]], np.ones(len(pk_idxs[i])) * 100, 'o', ms=7, color=colors[i],
                       alpha=.8, mec='k', mew=1.5, rasterized=True)

    calls_ax.plot(all_ch_peak_times, np.ones(len(all_ch_peak_times)) * 150, 'o', ms=10, color='gray',
                  alpha=.8, mec='k', mew=1.5, rasterized=True)

    pass


def get_calls_across_channels(single_filename, run_window_width=0.05, step_quotient=10, ch_jitter_th=0.005,
                              dr=50, plot_spec=False, debug_plot=False):
    """

    Parameters
    ----------
    single_filename: str.
    String with the full path and name of the file from which calls should be extracted from
    run_window_width: float.
        Width of the window (in seconds) that runs through the mean powers in search for calls
    step_quotient: int.
        Quotient specifying the fraction by which run_window_width is divided,
        thereby setting the step_size
    ch_jitter_th: float.
        The threshold below which calls are considered repetitions (i.e. multiple detections)
        in several channels
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
    specs, spec_params, pk_arrays = load_all_channels(single_filename, ret_call_pks=True)

    # The problem across channels is that there is a delay in the call from one mike to the other and
    # this results in double detections which are time-shifted.
    # I therefore need to decide on which channel to stay for a fixed period of time (running-windows).
    # The channel that registers higher amplitude values at call_times is the channel to be taken into account.

    av_pow = [np.mean(e, axis=0) for e in specs]
    norm_pow = [e - np.mean(e) for e in av_pow]  # Norm, because each mike has a different gain and therefore power
    time_sr = 1. / np.diff(spec_params[0][:2])[0]  # time resolution of the spectrogram

    # find window_size in index
    idx_window_width = int(np.floor(time_sr * run_window_width / 1.))
    step_size = idx_window_width // step_quotient  # step_size is 0.05s for a run_window_width of 0.2s
    last_idx = len(spec_params[0]) - idx_window_width
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
            c_callTimes = spec_params[0][win_peak_idxs[channel_w_highest_pow]]
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
        spec_fig, ch_axs, all_calls_ax = plot_multiCH_spectrogram(specs, spec_params, single_filename,
                                                                  dyn_range=dr, ret_fig_chAxs_and_callAx=True)
        plot_calls_in_spectrogram(spec_fig, ch_axs, pk_arrays, spec_params[0], all_calls_ax, call_times)

    if debug_plot:  # plot the normed powers for debugging
        fig, ax = plt.subplots()
        colors = ['purple', 'cornflowerblue', 'forestgreen', 'darkred']
        for i in np.arange(len(specs)):
            ax.plot(spec_params[0], norm_pow[i], color=colors[i], alpha=.8)
            ax.plot(spec_params[0][pk_arrays[i]], np.ones(len(pk_arrays[i])) * 1 + 1 * i, 'o', ms=7, color=colors[i],
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

    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xticklabels(['Call Duration', 'Fmax', 'Fpk', 'Fmin'], fontsize=fs+4)
    ax1.set_ylabel('Duration [ms]', fontsize=fs+4)
    ax2.set_ylabel('Frequency [kHz]', fontsize=fs+4)
    ax2.set_title('n = %.i' % len(cp_dict['call_number']), fontsize=fs+4)

    ax1.set_yticks(np.arange(0., 3.5, 0.5))
    ax2.set_yticks(np.arange(60., 220., 20))

    for ax in [ax1, ax2]:
        ax.tick_params(labelsize=fs+3)

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
