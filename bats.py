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

    def __init__(self, file_path, f_resolution=2**9, overlap_frac=0.7, dynamic_range=50, pcTape_rec=False,
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

    def compute_spectrogram(self):
        if not self.data_loaded:
            self.load_data()

        from thunderfish.powerspectrum import nfft
        n_nfft = nfft(self.sampling_rate, self.freq_resolution)
        self.spec_mat, self.f, self.t = mlab.specgram(self.recording_trace, NFFT=n_nfft, Fs=self.sampling_rate,
                                                      noverlap=int(n_nfft * self.overlap_frac))
        self.spectrogram_computed = True

        pass

    def plot_spectrogram(self, dec_mat=None, spec_mat=None, f_arr=None, t_arr=None, in_kHz=True, adjust_to_max_db=True,
                         ret_fig_and_ax=False, showit=True):

        if spec_mat is None and dec_mat is None:
            spec_mat = self.spec_mat
            dec_mat = decibel(spec_mat)

        elif spec_mat is not None and dec_mat is None:
            dec_mat = decibel(spec_mat)

        elif not self.spectrogram_computed:
            self.compute_spectrogram()

        if adjust_to_max_db:
            # set dynamic range

            ampl_max = np.nanmax(
                dec_mat)  # define maximum; use nanmax, because decibel function may contain NaN values
            dec_mat -= ampl_max + 1e-20  # subtract maximum so that the maximum value is set to lim x--> -0
            dec_mat[dec_mat < -self.dynamic_range] = - self.dynamic_range

            # Fix NaNs issue
            if True in np.isnan(dec_mat):
                dec_mat[np.isnan(dec_mat)] = - self.dynamic_range
        else:
            dec_mat = spec_mat

        if f_arr is None:
            f_arr = self.f

        if t_arr is None:
            t_arr = self.t

        if in_kHz:
            hz_fac = 1000
        else:
            hz_fac = 1

        inch_factor = 2.54
        fs = 20
        fig, ax = plt.subplots(figsize=(56. / inch_factor, 30. / inch_factor))
        im = ax.imshow(dec_mat, cmap='jet',
                       extent=[t_arr[0], t_arr[-1],
                               int(f_arr[0])/hz_fac, int(f_arr[-1])/hz_fac],  # divide by 1000 for kHz
                       aspect='auto', interpolation='hanning', origin='lower', alpha=0.7, vmin=-self.dynamic_range,
                       vmax=0.)

        cb = fig.colorbar(im)

        cb.set_label('dB', fontsize=fs)
        ax.set_ylabel('Frequency [kHz]', fontsize=fs)
        ax.set_xlabel('Time [sec]', fontsize=fs)
        ax.tick_params(labelsize=fs-1)

        # ToDo: Plot the soundwave underneath the spectrogram!!

        self.spectrogram_plotted = True

        if ret_fig_and_ax:
            return fig, ax
        else:
            pass

        if showit:
            plt.show()
        else:
            pass

    def detect_calls(self, det_range=(50000, 150000), th_between_calls=0.004, plot_debug=False,
                     plot_in_spec=False, save_spec_w_calls=False):

        # Get an average over all frequency channels within detection range
        av_power = np.mean(self.spec_mat[np.logical_and(self.f > det_range[0], self.f < det_range[1])], axis=0)

        th = np.min(av_power)  # THIS THRESHOLD ROCKS YOUR PANTS! for more detections, increase f_res. 2^7 or 2^8
        if th <= 0:  # Fix cases where th <= 0
            th = np.mean(av_power)
        peaks, _ = detect_peaks(av_power, th)  # Use thunderfish's peak-trough algorithm

        # clean pks that might be echoes
        below_t_th = np.diff(self.t[peaks]) < th_between_calls

        if len(np.where(below_t_th)[0]) == 0:
            cleaned_peaks = peaks
        else:
            cleaned_peaks = np.delete(peaks, np.where(below_t_th)[0])

        if plot_debug:
            fig, ax = plt.subplots()
            ax.plot(self.t, av_power)
            ax.plot(self.t[cleaned_peaks], np.ones(len(cleaned_peaks)) * np.max(av_power), 'o', ms=20, color='darkred', alpha=.8,
                    mec='k', mew=3)
            ax.plot([self.t[0], self.t[-1]], [th, th], '--k', lw=2.5)
            # plt.show()

        if plot_in_spec:
            spec_fig, spec_ax = self.plot_spectrogram(spec_mat=self.spec_mat, f_arr=self.f, t_arr=self.t, ret_fig_and_ax=True, showit=False)
            spec_ax.plot(self.t[cleaned_peaks], np.ones(len(cleaned_peaks))*80, 'o', ms=20,  # plots the detection at 80kHz
                         color='darkred', alpha=.8, mec='k', mew=3)
            spec_fig.suptitle(self.file_name.split('.')[0])
            if save_spec_w_calls:
                spec_fig.savefig('test_result/detected_calls/' + self.file_name.split('.')[0] + '.pdf')

        return av_power, cleaned_peaks


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
        calls, chOfCall = get_calls_across_channels(all_recs, run_window_width=0.05, step_quotient=10, plot_spec=True)
        chOfCall += 1  # set the channel name same as the filename

        # Compute the Pulse-Intervals:
        from call_intervals import get_CI_and_call_bouts, plot_call_bout_vs_CI
        bout_calls, bout_diffs = get_CI_and_call_bouts(calls)

        rec_dict = {enu+1: Batspy(rec, f_resolution=2**9, overlap_frac=.70, dynamic_range=70)
                    for enu, rec in enumerate(all_recs)}
        [rec_dict[e].load_data() for e in rec_dict.keys()]  # load the data in all channels

        # Goal now is to create small windows for each call
        # make a time array with the sampling rate
        time = np.arange(0, len(rec_dict[1].recording_trace) /
                         rec_dict[1].sampling_rate, 1/rec_dict[1].sampling_rate)
        window_width = 0.010  # in seconds

        call_dict = {'cb': [], 'ce': [], 'fb': [], 'fe': [], 'pf': [], 'call_number': []}
        print('\nCalls extracted, proceeding to loop through %.i detected calls...\n' % len(calls))

        for c_call in range(len(calls)):
            c_ch = chOfCall[c_call]
            nfft = 2 ** 8
            call_w_idx = np.logical_and(time >= calls[c_call] - window_width / 2.,
                                        time <= calls[c_call] + window_width / 2.)
            trace = rec_dict[c_ch].recording_trace[call_w_idx]

            s, f, t = mlab.specgram(trace, Fs=rec_dict[c_ch].sampling_rate, NFFT=nfft,
                                    noverlap=int(0.8 * nfft))  # Compute a high-res spectrogram of the window

            dec_spec = decibel(s)

            call_freq_range = (50000, 250000)
            filtered_spec = dec_spec[np.logical_and(f > call_freq_range[0], f < call_freq_range[1])]
            freqs_of_filtspec = np.linspace(call_freq_range[0], call_freq_range[-1], np.shape(filtered_spec)[0])

            # measure noise floor
            noiseEdge = int(np.floor(0.002 / np.diff(t)[0]))
            noise_floor = np.max(np.hstack((filtered_spec[:, :noiseEdge], filtered_spec[:, -noiseEdge:]))) + 2

            lowest_decibel = noise_floor

            # get peak frequency
            peak_f_idx = np.unravel_index(filtered_spec.argmax(),
                                          filtered_spec.shape)

            # ToDo: Make a function out of this in order to avoid code copy-paste
            left_from_peak = np.arange(peak_f_idx[1] - 1, -1, -1, dtype=int)
            right_from_pk = np.arange(peak_f_idx[1] + 1, len(t), dtype=int)

            mainHarmonicTrace = []
            db_th = 12.0
            f_tol_th = 40000  # in Hz
            t_tol_th = 0.0012  # in s

            freq_tolerance = np.where(np.cumsum(np.diff(freqs_of_filtspec)) > f_tol_th)[0][0]
            time_tolerance = np.where(np.cumsum(np.diff(t)) > t_tol_th)[0][0]

            # first start from peak to right
            f_ref = peak_f_idx[0]
            t_ref = peak_f_idx[1]
            mainHarmonicTrace.append([peak_f_idx[0], peak_f_idx[1]])
            for ri in right_from_pk:
                pi, _ = detect_peaks(filtered_spec[:, ri], db_th)
                pi = pi[filtered_spec[pi, ri] > lowest_decibel]

                if len(pi) > 0:
                    curr_f = pi[np.argmin(np.abs(f_ref - pi))]
                    if np.abs(ri - t_ref) > time_tolerance or np.abs(curr_f - f_ref) > freq_tolerance \
                            or f_ref - curr_f < 0:
                        continue
                    else:
                        mainHarmonicTrace.append([curr_f, ri])
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
                    if np.abs(li - t_ref) > time_tolerance or np.abs(curr_f - f_ref) > freq_tolerance \
                            or curr_f - f_ref < 0:
                        continue
                    else:
                        mainHarmonicTrace.insert(0, [curr_f, li])
                        f_ref = curr_f
                        t_ref = li
                else:
                    continue

            mainHarmonicTrace = np.array(mainHarmonicTrace)

            if np.abs(noise_floor - filtered_spec[peak_f_idx]) > db_th and \
                                    (t[mainHarmonicTrace[-1][1]] - t[mainHarmonicTrace[0][1]]) * 1000. > 1.2:
                call_dict['call_number'].append(c_call)
                call_dict['cb'].append(t[mainHarmonicTrace[0][1]])
                call_dict['ce'].append(t[mainHarmonicTrace[-1][1]])
                call_dict['fb'].append(freqs_of_filtspec[mainHarmonicTrace[0][0]])
                call_dict['fe'].append(freqs_of_filtspec[mainHarmonicTrace[-1][0]])
                call_dict['pf'].append(freqs_of_filtspec[peak_f_idx[0]])

        # Dictionary with call parameters should be filled here
        call_dict = {e: np.array(call_dict[e]) for e in call_dict.keys()}
        from multiCH import plot_call_parameter_distributions
        plot_call_parameter_distributions(call_dict, showit=False)

        embed()
        quit()

        fs = 18
        inch_factor = 2.54
        fig, ax = plt.subplots(figsize=(20./inch_factor, 20./inch_factor))

        all_diffs = np.hstack(bout_diffs)[np.logical_and(np.hstack(bout_diffs) > 0.008, np.hstack(bout_diffs) < 0.045)]
        all_diffs *= 1000.
        binw = 1
        bins = np.arange(0, 45+binw, binw)
        ax.hist(all_diffs, bins=bins-binw/2., color='gray', edgecolor='k', alpha=.9)

        fig, ax = plt.subplots(figsize=(40./inch_factor, 20./inch_factor))
        ax.plot(bout_calls[0][17:], bout_diffs[0][17:]*1000., 'o-', color='gray', ms=17, mec='k', mew=2, lw=2, alpha=0.8)

        ax.set_xlabel('Time until catch [s]', fontsize=fs)
        ax.set_ylabel('Call Interval [ms]', fontsize=fs)

        ax.tick_params(labelsize=fs-1)
        fig.tight_layout()
        plt.show()


        embed()
        quit()


        plot_call_bout_vs_CI(bout_calls, bout_diffs)

        plt.show()

    # Analyze SingleChannel
    elif rec_type == 's':

        bat = Batspy(recording, f_resolution=2**9, overlap_frac=.70, dynamic_range=50, pcTape_rec=False)  # 2^7 = 128
        bat.compute_spectrogram()
        # bat.plot_spectrogram(showit=False)
        pows, pks = bat.detect_calls(det_range=(80000, 150000), plot_in_spec=True)
        embed()
        # plt.show()
        quit()

        # ToDo: Need to improve the basic call detection algorithm!
        average_power, peaks = bat.detect_calls(det_range=(100000, 150000), plot_in_spec=False, plot_debug=False)

        # Goal now is to create small windows for each call
        # make a time array with the sampling rate
        time = np.arange(0, len(bat.recording_trace) / bat.sampling_rate, 1/bat.sampling_rate)
        window_width = 0.010  # in seconds
        # now the call windows
        call_windows = [bat.recording_trace[np.logical_and(time >= bat.t[e]-window_width/2.,
                                                           time <= bat.t[e]+window_width/2.)]
                        for e in peaks]

        call_dict = {'cb': [], 'ce': [], 'fb': [], 'fe': [], 'pf': [], 'call_number': []}

        for c_call in range(len(call_windows)):  # loop through the windows
            nfft = 2 ** 8
            s, f, t = mlab.specgram(call_windows[c_call], Fs=bat.sampling_rate,
                                    NFFT=nfft, noverlap=int(0.8 * nfft))  # Compute a high-res spectrogram of the window

            dec_spec = decibel(s)

            call_freq_range = (50000, 250000)
            filtered_spec = dec_spec[np.logical_and(f > call_freq_range[0], f < call_freq_range[1])]
            freqs_of_filtspec = np.linspace(call_freq_range[0], call_freq_range[-1], np.shape(filtered_spec)[0])

            # measure noise floor
            noiseEdge = int(np.floor(0.002 / np.diff(t)[0]))
            noise_floor = np.max(np.hstack((filtered_spec[:, :noiseEdge], filtered_spec[:, -noiseEdge:]))) + 2

            lowest_decibel = noise_floor

            # get peak frequency
            peak_f_idx = np.unravel_index(filtered_spec.argmax(),
                                          filtered_spec.shape)

            # ToDo: Make a function out of this in order to avoid code copy-paste
            left_from_peak = np.arange(peak_f_idx[1]-1, -1, -1, dtype=int)
            right_from_pk = np.arange(peak_f_idx[1]+1, len(t), dtype=int)

            mainHarmonicTrace = []
            db_th = 12.0
            f_tol_th = 40000  # in Hz
            t_tol_th = 0.0012  # in s

            freq_tolerance = np.where(np.cumsum(np.diff(freqs_of_filtspec)) > f_tol_th)[0][0]
            time_tolerance = np.where(np.cumsum(np.diff(t)) > t_tol_th)[0][0]

            # first start from peak to right
            f_ref = peak_f_idx[0]
            t_ref = peak_f_idx[1]
            mainHarmonicTrace.append([peak_f_idx[0], peak_f_idx[1]])
            for ri in right_from_pk:
                pi, _ = detect_peaks(filtered_spec[:, ri], db_th)
                pi = pi[filtered_spec[pi, ri] > lowest_decibel]

                if len(pi) > 0:
                    curr_f = pi[np.argmin(np.abs(f_ref - pi))]
                    if np.abs(ri - t_ref) > time_tolerance or np.abs(curr_f - f_ref) > freq_tolerance \
                            or f_ref - curr_f < 0:
                        continue
                    else:
                        mainHarmonicTrace.append([curr_f, ri])
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
                        mainHarmonicTrace.insert(0, [curr_f, li])
                        f_ref = curr_f
                        t_ref = li
                else:
                    continue

            mainHarmonicTrace = np.array(mainHarmonicTrace)

            if np.abs(noise_floor - filtered_spec[peak_f_idx]) > db_th and \
                                    (t[mainHarmonicTrace[-1][1]] - t[mainHarmonicTrace[0][1]]) * 1000. > 1.2:
                call_dict['call_number'].append(c_call)
                call_dict['cb'].append(t[mainHarmonicTrace[0][1]])
                call_dict['ce'].append(t[mainHarmonicTrace[-1][1]])
                call_dict['fb'].append(freqs_of_filtspec[mainHarmonicTrace[0][0]])
                call_dict['fe'].append(freqs_of_filtspec[mainHarmonicTrace[-1][0]])
                call_dict['pf'].append(freqs_of_filtspec[peak_f_idx[0]])

                # if (t[mainHarmonicTrace[-1][1]] - t[mainHarmonicTrace[0][1]]) * 1000. > 2.5:  # filter for calls longer than 2.5s
                fig, ax = bat.plot_spectrogram(dec_mat=filtered_spec, f_arr=freqs_of_filtspec, t_arr=t, ret_fig_and_ax=True)
                ax.plot(t[mainHarmonicTrace[:, 1]], freqs_of_filtspec[mainHarmonicTrace[:, 0]]/1000.,
                        'o', ms=12, color='None', mew=3, mec='k', alpha=0.7)
                ax.plot(t[peak_f_idx[1]], freqs_of_filtspec[peak_f_idx[0]] / 1000, 'o', ms=15, color='None', mew=4, mec='purple', alpha=0.8)
                ax.set_title('call # %i' % c_call, fontsize=20)

                import os
                save_path = '../../data/temp_batspy/' + '/'.join(bat.file_path.split('/')[5:-1]) +\
                            '/' + bat.file_name.split('.')[0] + '/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                fig.savefig(save_path + 'fig_' + str(c_call).zfill(4) + '.pdf')
                plt.close(fig)

        # Create figure of call parameters
        call_dict = {e:  np.array(call_dict[e]) for e in call_dict.keys()}
        from multiCH import plot_call_parameter_distributions
        plot_call_parameter_distributions(call_dict)

        embed()
        quit()
        print('\nDONE!')
