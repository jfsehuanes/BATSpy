import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from sklearn.linear_model import LinearRegression as linreg
from thunderfish.dataloader import load_data
from thunderfish.powerspectrum import spectrogram, decibel
from thunderfish.eventdetection import detect_peaks, percentile_threshold
from thunderfish.harmonicgroups import harmonic_groups
from thunderfish.powerspectrum import psd

from IPython import embed


class Batspy:

    def __init__(self, file_path, f_resolution=2**9, overlap_frac=0.7, dynamic_range=70, pcTape_rec=False,
                 multiCH=False):
        self.file_path = file_path
        self.file_name = file_path.split('/')[-1]
        self.freq_resolution = f_resolution  # in Hz
        self.overlap_frac = overlap_frac  # Overlap fraction of the FFT windows
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

    def plot_spectrogram(self, spec_mat, f_arr, t_arr, in_kHz=True, ret_fig_and_ax=False):
        if not self.spectrogram_computed:
            self.compute_spectrogram()

        if in_kHz:
            hz_fac = 1000
        else:
            hz_fac = 1

        inch_factor = 2.54
        fs = 14
        fig, ax = plt.subplots(figsize=(56. / inch_factor, 30. / inch_factor))
        im = ax.imshow(spec_mat, cmap='jet',
                       extent=[t_arr[0], t_arr[-1],
                               int(f_arr[0])/hz_fac, int(f_arr[-1])/hz_fac],  # divide by 1000 for kHz
                       vmin=-100.0, vmax=-50.0,
                       aspect='auto', origin='lower', alpha=0.7)

        cb = fig.colorbar(im)

        cb.set_label('dB', fontsize=fs)
        ax.set_ylabel('Frequency [kHz]', fontsize=fs)
        ax.set_xlabel('Time [sec]', fontsize=fs)
        self.spectrogram_plotted = True

        if ret_fig_and_ax:
            return fig, ax
        else:
            pass

    def detect_calls(self, strict_th=False, det_range=(50000, 150000), plot_debug=False,
                     plot_in_spec=False, save_spec_w_calls=False):

        # Get an average over all frequency channels within detection range
        av_power = np.mean(self.spec_mat[np.logical_and(self.f > det_range[0], self.f < det_range[1])], axis=0)

        if strict_th:  # either search for really good quality calls or else for just a rough detection
            th = np.percentile(av_power, 99)
        else:
            th = np.min(av_power)  # THIS THRESHOLD ROCKS YOUR PANTS! for more detections, increase f_res. 2^7 or 2^8

        # Fix cases where th <= 0
        if th <= 0:
            th = np.mean(av_power)
        peaks, troughs = detect_peaks(av_power, th)  # Use thunderfish's peak-trough algorithm

        if plot_debug:
            fig, ax = plt.subplots()
            ax.plot(self.t, av_power)
            ax.plot(self.t[peaks], np.ones(len(peaks)) * np.max(av_power), 'o', ms=20, color='darkred', alpha=.8,
                    mec='k', mew=3)
            ax.plot([self.t[0], self.t[-1]], [th, th], '--k', lw=2.5)
            # plt.show()

        if plot_in_spec:
            spec_fig, spec_ax = self.plot_spectrogram(self.plt_spec, self.f, self.t, ret_fig_and_ax=True)
            spec_ax.plot(self.t[peaks], np.ones(len(peaks))*80, 'o', ms=20,  # plots the detection at 80kHz
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
        from multiCH import get_all_ch, get_calls_across_channels

        # Get all the channels corresponding to the input file
        all_recs = get_all_ch(recording)
        # Get the calls
        calls = get_calls_across_channels(all_recs, run_window_width=0.05, step_quotient=10, plot_spec=True)

        # Compute the Pulse-Intervals:
        from call_intervals import get_CI_and_call_bouts, plot_call_bout_vs_CI
        bout_calls, bout_diffs = get_CI_and_call_bouts(calls)
        plot_call_bout_vs_CI(bout_calls, bout_diffs)

        plt.show()

    # Analyze SingleChannel
    elif rec_type == 's':

        from helper_functions import extract_peak_and_th_crossings_from_cumhist
        # ToDo change fresolution to NNFT!!!
        bat = Batspy(recording, f_resolution=2**9, overlap_frac=.70, dynamic_range=70)  # 2^7 = 128
        bat.compute_spectrogram()
        average_power, peaks, _ = bat.detect_calls(strict_th=False, plot_in_spec=False)
        # ToDo: Make a better noise analysis and adapt the call-feature-detection-thresholds, so that this also works
        # ToDo: when strict_th=False!

        # Goal now is to create small windows for each call
        # make a time array with the sampling rate
        time = np.arange(0, len(bat.recording_trace) / bat.sampling_rate, 1/bat.sampling_rate)
        window_width = 0.010  # in seconds
        # now the call windows
        call_windows = [bat.recording_trace[np.logical_and(time >= bat.t[e]-window_width/2.,
                                                           time <= bat.t[e]+window_width/2.)]
                        for e in peaks]

        for c_call in np.arange(len(call_windows)):  # loop through the windows
            nfft = 2 ** 8
            s, f, t = mlab.specgram(call_windows[c_call], Fs=bat.sampling_rate, NFFT=nfft, noverlap=int(0.8 * nfft))  # Compute a high-res spectrogram of the window

            dec_spec = decibel(s)

            call_freq_range = (50000, 250000)
            filtered_spec = dec_spec[np.logical_and(f > call_freq_range[0], f < call_freq_range[1])]
            freqs_of_filtspec = np.linspace(call_freq_range[0], call_freq_range[-1], np.shape(filtered_spec)[0])

            # measure noise floor
            noise_floor = np.max(filtered_spec[:, :10])
            lowest_decibel = noise_floor

            # get peak frequency
            peak_f_idx = np.unravel_index(filtered_spec.argmax(),
                                          filtered_spec.shape)

            # ToDo: Make a function out of this in order to avoid code copy-paste
            left_from_peak = np.arange(peak_f_idx[1]-1, -1, -1, dtype=int)
            right_from_pk = np.arange(peak_f_idx[1]+1, len(t), dtype=int)

            pre_call_trace = []
            db_th = 20.0
            f_tol_th = 15000  # in Hz
            t_tol_th = 0.0004  # in s

            freq_tolerance = np.where(np.cumsum(np.diff(freqs_of_filtspec)) > f_tol_th)[0][0]
            time_tolerance = np.where(np.cumsum(np.diff(t)) > t_tol_th)[0][0]

            # first start from peak to right
            f_ref = peak_f_idx[0]
            t_ref = peak_f_idx[1]
            pre_call_trace.append([peak_f_idx[0], peak_f_idx[1]])
            for ri in right_from_pk:
                pi, _ = detect_peaks(filtered_spec[:, ri], db_th)
                pi = pi[filtered_spec[pi, ri] > lowest_decibel]

                if len(pi) > 0:
                    curr_f = pi[np.argmin(np.abs(f_ref - pi))]
                    if np.abs(ri - t_ref) > time_tolerance or np.abs(curr_f - f_ref) > freq_tolerance \
                            or f_ref - curr_f < 0:
                        continue
                    else:
                        pre_call_trace.append([curr_f, ri])
                        f_ref = curr_f
                        t_ref = ri
                else:
                    continue

            # Now from peak to left
            f_ref = peak_f_idx[0]
            t_ref = peak_f_idx[1]
            for li in left_from_peak:
                pi, _ = detect_peaks(filtered_spec[:, li], db_th)
                pi = pi[filtered_spec[pi, li] > lowest_decibel]

                if len(pi) > 0:
                    curr_f = pi[np.argmin(np.abs(f_ref - pi))]
                    if np.abs(li - t_ref) > time_tolerance or np.abs(curr_f - f_ref) > freq_tolerance\
                            or curr_f - f_ref < 0:
                        continue
                    else:
                        pre_call_trace.insert(0, [curr_f, li])
                        f_ref = curr_f
                        t_ref = li
                else:
                    continue

            pre_call_trace = np.array(pre_call_trace)

            fig, ax = bat.plot_spectrogram(filtered_spec, freqs_of_filtspec, t, ret_fig_and_ax=True)
            ax.plot(t[pre_call_trace[:, 1]], freqs_of_filtspec[pre_call_trace[:, 0]]/1000.,
                    'o', ms=12, color='None', mew=3, mec='k', alpha=0.7)
            ax.plot(t[peak_f_idx[1]], freqs_of_filtspec[peak_f_idx[0]] / 1000, 'o', ms=15, color='None', mew=4, mec='purple', alpha=0.8)
            ax.set_title('call # %i' % c_call)

            if c_call == 60:
                embed()
                quit()

            # import os
            # save_path = '../../data/temp_batspy/' + '/'.join(bat.file_path.split('/')[5:-1]) +\
            #             '/' + bat.file_name.split('.')[0] + '/'
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            #
            # fig.savefig(save_path + 'fig_' + str(c_call).zfill(4) + '.pdf')
            # plt.close(fig)
        #
        # print('\nDONE!')
