import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib.pyplot as plt

from thunderfish.dataloader import load_data
from thunderfish.powerspectrum import spectrogram, decibel
from thunderfish.eventdetection import detect_peaks, threshold_crossings

from IPython import embed


class Batspy:

    def __init__(self, file_path, f_resolution=2**7, overlap_frac=0.64, dynamic_range=90, pcTape_rec=False,
                 multiCH=False):
        self.file_path = file_path
        self.file_name = file_path.split('/')[-1]
        self.freq_resolution = f_resolution  # in Hz
        self.overlap_frac = overlap_frac  # Overlap fraction of the NFFT windows
        self.dynamic_range = dynamic_range  # in dB
        self.pcTape_rec = pcTape_rec  # Was this file recorded by PC-Tape?
        self.multiCH = multiCH  # Does this file was recorded with other channels simultaneously?

        # Flow control booleans
        self.data_loaded = False
        self.spectrogram_computed = False
        self.spectrogram_plotted = False

    def load_data(self):
        dat, sr, u = load_data(self.file_path)
        self.recording_trace = dat.squeeze()

        if self.pcTape_rec:  # This fixes PC-Tape's bug that writes 1/10 of the samplingrate in the header of the .wav file
            self.sampling_rate = sr * 10.
        else:
            self.sampling_rate = sr

        self.data_loaded = True
        pass

    def compute_spectrogram(self, plotit=False, ret=False):
        if not self.data_loaded:
            self.load_data()
        self.spec_mat, self.f, self.t = spectrogram(self.recording_trace, self.sampling_rate,
                                                fresolution=self.freq_resolution, overlap_frac=self.overlap_frac)

        # set dynamic range
        dec_spec = decibel(self.spec_mat)
        ampl_max = np.nanmax(dec_spec)  # define maximum; use nanmax, because decibel function may contain NaN values
        dec_spec -= ampl_max + 1e-20  # subtract maximum so that the maximum value is set to lim x--> -0
        dec_spec[dec_spec < -self.dynamic_range] = -self.dynamic_range

        # Fix NaNs issue
        if True in np.isnan(dec_spec):
            dec_spec[np.isnan(dec_spec)] = - self.dynamic_range

        self.plt_spec = dec_spec

        self.spectrogram_computed = True

        if plotit:
            self.plot_spectrogram()
        pass

    def plot_spectrogram(self, ret_fig_and_ax=False):
        if not self.spectrogram_computed:
            self.compute_spectrogram()

        inch_factor = 2.54
        fs = 14
        fig, ax = plt.subplots(figsize=(56. / inch_factor, 30. / inch_factor))
        im = ax.imshow(self.plt_spec, cmap='jet', extent=[self.t[0], self.t[-1], self.f[0], self.f[-1]], aspect='auto',
                       origin='lower', alpha=0.7)
        # ToDo: Impossible to update the colorbar ticks for them to be multiples of 10!!!
        cb_ticks = np.arange(0, self.dynamic_range + 10, 10)

        cb = fig.colorbar(im)

        cb.set_label('dB', fontsize=fs)
        ax.set_ylabel('Frequency [Hz]', fontsize=fs)
        ax.set_xlabel('Time [sec]', fontsize=fs)
        self.spectrogram_plotted = True

        if ret_fig_and_ax:
            return fig, ax
        else:
            pass

    def detect_calls(self, det_range=(95000., 180000.), d_range_det_th=0.1, plot_debug=False,
                     plot_in_spec=False, save_spec_w_calls=False):

        if d_range_det_th > 1. or d_range_det_th < 0.:
            raise(ValueError("The detection threshold should be between 0 and 1"))

        # SET A PROPER THRESHOLD

        # Get an average over all frequency channels within detection range
        av_power = np.mean(self.spec_mat[np.logical_and(self.f > det_range[0], self.f < det_range[1])], axis=0)
        th = np.min(av_power)  # THIS THRESHOLD ROCKS YOUR PANTS! for more detections, increase f_res. 2^7 or 2^8

        # Fix cases where th <= 0
        if th <= 0:
            th = np.mean(av_power)
        peaks, troughs = detect_peaks(av_power, th)

        if plot_debug:
            fig, ax = plt.subplots()
            ax.plot(self.t, av_power)
            ax.plot(self.t[peaks], np.ones(len(peaks)) * np.max(av_power), 'o', ms=20, color='darkred', alpha=.8,
                    mec='k', mew=3)
            ax.plot([self.t[0], self.t[-1]], [th, th], '--k', lw=2.5)
            # plt.show()

        if plot_in_spec:
            spec_fig, spec_ax = self.plot_spectrogram(ret_fig_and_ax=True)
            spec_ax.plot(self.t[peaks], np.ones(len(peaks))*det_range[1] + (det_range[1]*0.001), 'o', ms=20,
                         color='darkred', alpha=.8, mec='k', mew=3)
            spec_fig.suptitle(self.file_name.split('.')[0])
            if save_spec_w_calls:
                spec_fig.savefig('test_result/detected_calls/' + self.file_name.split('.')[0] + '.pdf')

        return av_power, peaks, troughs


if __name__ == '__main__':

    import sys

    if len(sys.argv) != 3:
        print("ERROR\nPlease tell me the FilePath of the recording you wish to analyze as 1st argument and if it is"
              " a single recording ('s') or part of a multi-channel ('m') recording as second argument")
        quit()

    recording = sys.argv[1]
    rec_type = sys.argv[2]

    # Analyze MultiChannel
    if rec_type == 'm':
        from multiCH import get_all_ch, plot_multiCH_spectrogram

        dyn_range = 70
        
        all_recs = get_all_ch(recording)
        pk_arrays = []
        specs = []

        for rec_idx in np.arange(len(all_recs)):
            print('\nAnalyzing Channel ' + str(rec_idx+1) + ' of ' + str(len(all_recs)) + '...')
            bat = Batspy(all_recs[rec_idx], f_resolution=2**9, overlap_frac=.70, dynamic_range=dyn_range)
            bat.compute_spectrogram()
            specs.append(bat.plt_spec)
            _, p, _ = bat.detect_calls()
            pk_arrays.append(p)

        # The problem across channels is that there is a delay in the call from one mike to the other and
        # this results in double detections which are time-shifted.
        # Now I need to decide on which channel to stay and make some compromises from there on!

        # make a running average window. The channel that registers higher values within the channel is the
        # channel to be taken into account.

        av_pow = [np.mean(e, axis=0) for e in specs]
        norm_pow = [e - np.mean(e) for e in av_pow]

        pow_at_call = [av_pow[e][pk_arrays[e]] for e in np.arange(len(pk_arrays))]
        time_at_call = [bat.t[pk_arrays[e]] for e in np.arange(len(pk_arrays))]
        run_wsize = 0.2  # in seconds
        time_sr = 1./np.diff(bat.t[:2])[0]

        # find window_size in index
        idx_window_width = int(np.floor(time_sr * run_wsize / 1.))
        step_size = idx_window_width//4
        last_idx = len(bat.t) - idx_window_width
        steps = np.arange(0, last_idx, step_size)

        channel_sequence = np.ones(len(steps))
        call_times = []

        for enu, step in enumerate(steps):

            # first we need the valid indices for the current window (for all channels, i.e shape=(n,4))
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
                call_times.append(bat.t[win_peak_idxs[channel_w_highest_pow]])

        # Now there are double detections for the same call when transitioning from one channel to the other.
        # For this I need to make an average of the time call for the case with double detections.
        call_times = np.unique(np.hstack(call_times))  # Use unique to remove same call detected in several windows
        th_2_calls = 0.004
        reps_idx = np.where(np.diff(call_times) <= th_2_calls)[0]  # array with repetition indices
        replacements = np.array([call_times[e] + (call_times[e+1] - call_times[e]) / 2. for e in reps_idx])
        call_times[reps_idx] = replacements  # first replace with average
        call_times = np.delete(call_times, reps_idx+1)  # then remove the extra call

        # plot the normed powers for debugging
        # colors = ['purple', 'cornflowerblue', 'forestgreen', 'darkred']
        # for i in np.arange(len(specs)):
        #     plt.plot(bat.t, norm_pow[i], color=colors[i], alpha=.8)
        #     plt.plot(bat.t[pk_arrays[i]], np.ones(len(pk_arrays[i])) * 1 + 1 * i, 'o', ms=20, color=colors[i],
        #              alpha=.8, mec='k', mew=3)
        # plt.plot(call_times, np.ones(len(call_times)) * 6, 'o', ms=20, color='gray', alpha=.8, mec='k', mew=3)

        # plot a spectrogram that includes all channels!
        plot_multiCH_spectrogram(np.mean(specs, axis=0), bat.t, bat.f, pk_arrays, call_times, dynamic_range=dyn_range)

        plt.show()
        quit()



    elif rec_type == 's':

        # For Myrna's data
        # mouse = Batspy(recording, f_resolution=2**6, overlap_frac=0.4)
        # mouse.compute_spectrogram()
        # mouse.detect_calls(det_range=(30000., 100000.), plot_debug=True, plot_in_spec=True)
        # plt.show()
        # quit()
        
        bat = Batspy(recording, f_resolution=2**9, overlap_frac=.70, dynamic_range=70)  # 2^7 = 128
        bat.compute_spectrogram()
        bat.detect_calls(plot_in_spec=True)
        embed()
        quit()
        plt.show()
        quit()
    

    # import glob
    # wavefiles = np.sort(glob.glob('../../data/fixed_files/*.wav'))

    # for e, wf in enumerate(wavefiles):
    #     print("\nAnalyzing file %i from %i\n" % (e+1, len(wavefiles)))
    #     bat = Batspy(wf, dynamic_range=70)
    #     bat.compute_spectogram()
    #     bat.detect_calls(plot_debug=True)

    #     plt.show()
    # quit()

    # # Get the data
    # recording1 = 'test_data/natalusTumidirostris0024.wav'
    # recording2 = 'test_result/natalusTumidirostris0045_fix.wav'
    # recording3 = '../../data/diana/0409 Tyroptera tricolor0061 + mit isolation call.wav'
    # stimulus = 'test_result/stim.wav'
    #
    # bat1 = Batspy(recording1, pcTape_rec=True)
    # bat1.compute_spectogram()
    # bat1.detect_calls(plot_in_spec=True, save_spec_w_calls=True)

    # bat2 = Batspy(recording2, pcTape_rec=False)
    # bat2.compute_spectogram()
    # bat2.detect_calls(plot_debug=True, plot_in_spec=True)
    #
    # stim = Batspy(stimulus, pcTape_rec=False)
    # stim.compute_spectogram()
    # stim.detect_calls()
    #
    #
    # plt.show()
    # quit()
