import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pandas as pd

from sklearn.linear_model import LinearRegression as linreg
from thunderfish.dataloader import load_data
from thunderfish.powerspectrum import spectrogram, decibel
from thunderfish.eventdetection import detect_peaks, percentile_threshold
from thunderfish.harmonics import harmonic_groups
from thunderfish.powerspectrum import psd

from call_parameters import call_window

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
                         ret_fig_and_ax=False, fig_input=None, showit=True):

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

        if fig_input is None:
            fig = plt.figure(constrained_layout=True, figsize=(56. / inch_factor, 30. / inch_factor))
        else:
            fig = fig_input
        gs = fig.add_gridspec(2, 3, height_ratios=(4, 1), width_ratios=(4.85, 4.85, .3))
        ax0 = fig.add_subplot(gs[0, :-1])
        ax1 = fig.add_subplot(gs[1, :-1])
        ax2 = fig.add_subplot(gs[0:-1, -1])


        im = ax0.imshow(dec_mat, cmap='jet',
                       extent=[t_arr[0], t_arr[-1],
                               int(f_arr[0])/hz_fac, int(f_arr[-1])/hz_fac],  # divide by 1000 for kHz
                       aspect='auto', interpolation='hanning', origin='lower', alpha=0.7, vmin=-self.dynamic_range,
                       vmax=0.)

        cb = fig.colorbar(im, cax=ax2)

        cb.set_label('dB', fontsize=fs)
        ax0.set_ylabel('Frequency [kHz]', fontsize=fs+1)

        ax1.set_ylabel('Amplitude [a.u.]', fontsize=fs+1)
        ax1.set_xlabel('Time [sec]', fontsize=fs+1)

        for c_ax in [ax0, ax1, ax2]:
            c_ax.tick_params(labelsize=fs)

        # Plot the soundwave underneath the spectrogram
        ax1.set_facecolor('black')
        time_arr = np.arange(0, len(self.recording_trace)/self.sampling_rate, 1/self.sampling_rate)
        ax1.plot(time_arr, self.recording_trace, color='yellow', lw=2, rasterized=True)

        # Share the time axis of spectrogram and raw sound trace
        ax0.get_shared_x_axes().join(ax0, ax1)
        ax1.set_xlim(0, time_arr[-1])

        # Remove time xticks of the spectrogram
        ax0.xaxis.set_major_locator(plt.NullLocator())

        self.spectrogram_plotted = True

        if ret_fig_and_ax:
            return fig, (ax0, ax1)
        else:
            pass

        if showit:
            plt.show()
        else:
            pass

    def detect_calls(self, det_range=(50000, 180000), th_between_calls=0.004, plot_debug=False,
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
            ax.plot(self.t[cleaned_peaks], np.ones(len(cleaned_peaks)) * np.max(av_power), 'o', ms=10, color='darkred',
                    alpha=.8, mec='k', mew=1.5)
            ax.plot([self.t[0], self.t[-1]], [th, th], '--k', lw=2.5)
            # plt.show()

        if plot_in_spec:
            spec_fig, spec_ax = self.plot_spectrogram(spec_mat=self.spec_mat, f_arr=self.f, t_arr=self.t,
                                                      ret_fig_and_ax=True, showit=False)
            spec_ax = spec_ax[0]
            spec_ax.plot(self.t[cleaned_peaks], np.ones(len(cleaned_peaks))*80, 'o', ms=10,  # plots the detection at 80kHz
                         color='darkred', alpha=.8, mec='k', mew=1.5)
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
        from helper_functions import manualCallDetectionAdjustment

        # Get all the channels corresponding to the input file
        all_recs = get_all_ch(recording)
        # Get the calls
        calls, chOfCall = get_calls_across_channels(all_recs, run_window_width=0.05, step_quotient=10, f_res=2**9,
                                                    overlap=0.7, dr=70, plot_spec=False)
        chOfCall += 1  # set the channel name same as the filename

        # # Here to switch on the interactive window for detecting the calls and add them to a csv file
        # specFig = plt.gcf()  # plot_spec needs to be True in get_calls_across_channels() function.
        # manualCallDetectionAdjustment(specFig, calls, recording)

        # Here for individual call parameter extraction
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

        callDur = np.zeros(len(calls))
        freqBeg = np.zeros(len(calls))
        freqEnd = np.zeros(len(calls))
        peakFreq = np.zeros(len(calls))
        callsMask = np.zeros(len(calls))
        enu = 0

        for channel in set(chOfCall):
            # load data
            dat, sr, u = load_data(all_recs[channel-1])
            dat = np.hstack(dat)

            for callT in calls[chOfCall == channel]:

                print('analyzing call %i' % (enu+1))  # calls are not analyzed in order ;)

                # compute a high res spectrogram of a defined window length
                dur, fb, fe, pf = call_window(dat, sr, callT, plotDebug=True)

                # save the parameters
                callDur[enu] = dur
                freqBeg[enu] = fb
                freqEnd[enu] = fe
                peakFreq[enu] = pf
                callsMask[enu] = callT

                # save the debug figure
                fig = plt.gcf()
                fig.suptitle('__CALL#' + '{:03}'.format(enu + 1), fontsize=14)
                fig.savefig(
                    'tmp/plots/' + 'CALL#' + '{:03}'.format(enu + 1) + '.pdf')
                plt.close(fig)

                enu += 1

        # Reorder the arrays and create a csv
        path = 'tmp/call_params/'
        sortedInxs = np.argsort(callsMask)
        paramsdf = pd.DataFrame({'callTime': callsMask[sortedInxs], 'bch': chOfCall[sortedInxs],
                                 'callDur': callDur[sortedInxs], 'fBeg': freqBeg[sortedInxs],
                                 'fEnd': freqEnd[sortedInxs], 'pkfreq': peakFreq[sortedInxs]})
        paramsdf.to_csv(path_or_buf=path + '__'.join(recording.split('/')[2:]) + '.csv', index=False)
        print('CHE ACABOOO')
        exit()



        # Dictionary with call parameters should be filled here
        call_dict = {e: np.array(call_dict[e]) for e in call_dict.keys()}
        from multiCH import plot_call_parameter_distributions
        plot_call_parameter_distributions(call_dict, showit=True)
        plt.show()
        quit()

        # from helper_functions import save_pis_and_call_parameters
        # save_pis_and_call_parameters(all_diffs, call_dict, '../phd_figures/call_parameter_arrays/')
        quit()


    # Analyze SingleChannel
    elif rec_type == 's':

        from call_intervals import extract_pulse_sequence, save_ipi_sequence

        bat = Batspy(recording, f_resolution=2**10, overlap_frac=.90, dynamic_range=70, pcTape_rec=False)  # 2^7 = 128
        bat.compute_spectrogram()
        # bat.plot_spectrogram(showit=False)
        pows, pks = bat.detect_calls(det_range=(50000, 180000), plot_in_spec=True, plot_debug=False)

        # create the header for the csv
        r = '/'.join(recording.split('/')[-3:])
        shortHeader = '_'.join([r[5:22], 'ch', r[-19], 'rec', r[-6:-4]])

        plt.show()

        embed()
        quit()

        # save pulse intervals
        pulse_times = extract_pulse_sequence(bat.t[pks], (1.43, 2.92), to_add=[1.5286])
        save_ipi_sequence(pulse_times, 'approach', shortHeader)
        ipis = np.diff(pulse_times)
        embed()
        quit()
        # bat.plot_spectrogram()
        plt.show()
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
            db_th = 15.0
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
                # fig, ax = bat.plot_spectrogram(dec_mat=filtered_spec, f_arr=freqs_of_filtspec, t_arr=t, ret_fig_and_ax=True)
                # ax.plot(t[mainHarmonicTrace[:, 1]], freqs_of_filtspec[mainHarmonicTrace[:, 0]]/1000.,
                #         'o', ms=12, color='None', mew=3, mec='k', alpha=0.7)
                # ax.plot(t[peak_f_idx[1]], freqs_of_filtspec[peak_f_idx[0]] / 1000, 'o', ms=15, color='None', mew=4, mec='purple', alpha=0.8)
                # ax.set_title('call # %i' % c_call, fontsize=20)
                #
                # import os
                # save_path = '../../data/temp_batspy/' + '/'.join(bat.file_path.split('/')[5:-1]) +\
                #             '/' + bat.file_name.split('.')[0] + '/'
                # if not os.path.exists(save_path):
                #     os.makedirs(save_path)
                #
                # fig.savefig(save_path + 'fig_' + str(c_call).zfill(4) + '.pdf')
                # plt.close(fig)

        # Create figure of call parameters
        call_dict = {e:  np.array(call_dict[e]) for e in call_dict.keys()}
        from multiCH import plot_call_parameter_distributions
        plot_call_parameter_distributions(call_dict)

        # embed()
        plt.show()
        quit()
        print('\nDONE!')
