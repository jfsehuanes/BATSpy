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

            #filtered_spec[filtered_spec < lowest_decibel] = lowest_decibel

            fig, ax = bat.plot_spectrogram(filtered_spec, freqs_of_filtspec, t, ret_fig_and_ax=True)

            # get peak frequency
            peak_f_idx = np.unravel_index(filtered_spec.argmax(),
                                          filtered_spec.shape)

            # calculate the frequency distance between the previous and current pi.
            # join previous pi to the current pi with minimum distance. That way we join all the maxima
            # corresponding to the main harmonic of the call. Next step should get rid of noise artifacts
            # and find harmonics.

            db_th = 10.0
            for ti in range(len(t)):  # instead of looping through the whole call window, I should start with the peak and then left and right from it until I don't find any more peaks!
                
                pi, _ = detect_peaks(filtered_spec[:, ti], db_th)
                pi = pi[filtered_spec[pi, ti] > lowest_decibel]


                # peak_freq = freqs_of_filtspec[pi]
                # mi = np.argmin(np.abs((peak_freq - f_max)))
                # peak_freq[mi]
                ax.plot(np.zeros(len(pi))+t[ti], freqs_of_filtspec[pi]/1000.0, '.k', ms=20.0)

            if c_call == 10:
                embed()
                quit()

            continue


            '''# Extract noisy artifacts
            noise_wlength = len(t) // 5
            blanc_spec = np.hstack((dec_spec[:, :noise_wlength], dec_spec[:, -noise_wlength:]))
            noise_psd = np.mean(blanc_spec, axis=1)

            th_noise = percentile_threshold(noise_psd, thresh_fac=0.3, percentile=1.0)

            noise_pks, noise_tr = detect_peaks(noise_psd, threshold=th_noise)  # th is in dB
            # Kill-all peaks < x dB of max(noise_psd(noise_pk))
            mx_noise_pk = np.max(noise_psd[noise_pks])
            th_from_maxpk = 5  # in dB
            valid_noise_pks = noise_pks[noise_psd[noise_pks] > mx_noise_pk - th_from_maxpk]

            # ToDo: subtract the corresponding value of each peak detected in noise_pks in s (not in dec_spec),
            # ToDO: i.e. the noise bands of the artifacts! Then "re-decibel"
            noise_attenuation = 4  # in dB
            adjacent_th = 3000  # in Hz
            # number of slots left and right of each pk noise to be attenuated
            adj_noisefreq_slots = np.where(f - f[1] <= adjacent_th)[0][-1] - 1
            freq_ids_to_attn = np.unique([[e - adj_noisefreq_slots, e, e + adj_noisefreq_slots] \
                                          for e in noise_pks if e + adj_noisefreq_slots < len(f)])
            dec_spec[freq_ids_to_attn, :] = dec_spec[freq_ids_to_attn, :] - noise_attenuation'''

            from scipy.stats.mstats import gmean
            amu = np.mean(np.abs(filtered_spec), axis=0)
            gmu = gmean(np.abs(filtered_spec), axis=0)

            tonality = 1 - gmu/amu

            # ton_th = 0.02  # tonality threshold
            ton_th = np.percentile(tonality, 75)  # tonality threshold

            # get indices of call begin and end
            c_beg = peak_f_idx[1] - np.where(tonality[:peak_f_idx[1]][::-1] < ton_th)[0][0]
            c_end = np.where(tonality[peak_f_idx[1]:] < ton_th)[0][0] + peak_f_idx[1]

            # get indices of begin and end frequencies
            f_beg = np.argmax(filtered_spec[:, c_beg])
            f_end = np.argmax(filtered_spec[:, c_end])

            # peak-detector throughout the call in order to detect harmonics
            db_th = 20.
            absolut_th = -18

            # first define how many harmonics are present and create a empty lists accordingly
            p, _ = detect_peaks(filtered_spec[:, peak_f_idx[1]], db_th)
            cf_mat = [[], []]

            # loop from peak-freq to call-end
            # ToDo: Maybe set call end and call beginning boundaries to +-15ms from peak-freq
            rightsweep = np.arange(peak_f_idx[1], c_end+1)
            leftsweep = np.arange(peak_f_idx[1]-1, c_beg-1, -1)
            approx_call_ids = np.concatenate((leftsweep, rightsweep))

            for ti in approx_call_ids:
                cp, _ = detect_peaks(filtered_spec[:, ti], db_th)
                valid_pks_ids = np.where(filtered_spec[:, ti][cp] > absolut_th)[0]

                if len(valid_pks_ids) == 0:
                    continue
                else:
                    cf_mat[0].append(ti)
                    cf_mat[1].append(cp[valid_pks_ids])

                # if len(cp) == 3:
                #     fig1, ax1 = plt.subplots()
                #     ax1.plot(freqs_of_filtspec, filtered_spec[:, r])

                # for harm in np.arange(len(cp)):
                #     cf_lists[harm] = cp[harm]


            # fig1, ax1 = plt.subplots()
            # ax1.plot(t, tonality, 'o')
            # ax1.set_title('call # %i' % c_call)
            # ax1.plot([t[0], t[-1]], [ton_th, ton_th], '--k')

            # # peak detection in each time slot with th = abs([0dB - median(spec)] * 0.5)
            # db_th = 16
            # peakdet_th_in_mat = np.abs(np.max(blanc_spec) - np.median(blanc_spec) * 0.5)
            #
            # # discard all peaks of previous step where peak < th
            # ls_to_fill = []
            # for tw in np.arange(len(t)):
            #     peaks_per_window = detect_peaks(filtered_spec[:, tw], peakdet_th_in_mat)[0]
            #     filtr = np.where(filtered_spec[peaks_per_window, tw] > -db_th)[0]
            #     if len(filtr) == 0:
            #         continue
            #     elif len(filtr) > 0:
            #         [ls_to_fill.append([peaks_per_window[e], tw]) for e in filtr]
            #
            # abv_th_mat = np.vstack(ls_to_fill)
            #
            # # ToDo: make a linear regression with the peaks a few time slots left and right of peak_max. then walk
            # # ToDo: through the time-steps with a time and frequency threshold in order to define the call
            #
            # steps_from_peakf = 1
            # a = np.unique(abv_th_mat[:, 0])
            # reg_window_ids = a[(np.where(a == peak_f_idx[0]
            #                              )[0][0] - steps_from_peakf): (np.where(a == peak_f_idx[0]
            #                                                                           )[0][0]+steps_from_peakf)+1]
            # strt_idx = np.where(abv_th_mat == reg_window_ids[-1])[0][0]
            # end_idx = np.where(abv_th_mat == reg_window_ids[0])[0][-1]
            #
            # reg_times = t[abv_th_mat[strt_idx:end_idx+1][:, 1]]
            # reg_times = reg_times.reshape(-1, 1)
            # reg_freqs = freqs_of_filtspec[abv_th_mat[strt_idx:end_idx+1][:, 0]]
            #
            # regressor = linreg()
            # regressor.fit(reg_times, reg_freqs)
            #
            # # Get call start and end
            # steps = len(t)//5
            # end_idx = 0
            # start_idx = 0
            #
            # for step in np.arange(peak_f_idx[-1], peak_f_idx[-1]+steps):  # walk from peak-f to call-end
            #     c_db = filtered_spec[valid_peaks_p_w[step], step][0]
            #
            #     if len(c_db) == 0:
            #         continue
            #
            #     if c_db < -db_th:
            #         end_idx = step
            #         break
            #     else:
            #         continue
            #
            # for bstep in np.arange(peak_f_idx[-1], peak_f_idx[-1]-steps, -1):  # walk from peak-f to call-start
            #     c_db = filtered_spec[valid_peaks_p_w[bstep], bstep][0]
            #
            #     if len(c_db) == 0:
            #         continue
            #
            #     if c_db < -db_th:
            #         start_idx = bstep
            #         break
            #     else:
            #         continue
            #
            # if np.logical_or(start_idx == 0, end_idx == 0):
            #     print('\nWARNING! There was a problem detecting call #%.0f.'
            #           'Proceeding analysis without including this call!\n' % tw)
            #     continue
            #
            # call_boundaries = np.array([start_idx, end_idx])

            fig, ax = bat.plot_spectrogram(filtered_spec, freqs_of_filtspec, t, ret_fig_and_ax=True)
            for i in np.arange(len(cf_mat[0])):
                ax.plot(t[[cf_mat[0][i]] * len(cf_mat[1][i])], freqs_of_filtspec[cf_mat[1][i]] / 1000., '.k', ms=10,
                        alpha=0.6)

            # ax.plot([t[c_beg], t[c_end]], np.array([freqs_of_filtspec[f_beg], freqs_of_filtspec[f_end]]) / 1000., '.k', alpha=0.8)
            ax.set_title('call # %i' % c_call)

            # ax.plot(t, freqs_of_filtspec[peaks_per_window] / 1000., 'o', color='gray', ms=8, mec='k', mew=2, alpha=.4)
            # ax.plot(t[call_boundaries], freqs_of_filtspec[peaks_per_window[call_boundaries]] / 1000., 'o', color='navy', ms=15,
            #         mec='k', mew=2, alpha=.7)
            # ax.plot(t[abv_th_mat[:, 1]], freqs_of_filtspec[abv_th_mat[:, 0]] / 1000., 'o', color='gray', ms=8, mec='k',
            #         mew=2, alpha=.4)
            # ax.plot(t[peak_f_idx[-1]], freqs_of_filtspec[peak_f_idx[0]] / 1000., 'o', color='navy', ms=15,
            #         mec='k', mew=2, alpha=.7)
            #
            # ax.plot(t, regressor.predict(t.reshape(-1, 1))/1000., '--k', lw=2, alpha=0.7)

            if c_call == 50:
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
