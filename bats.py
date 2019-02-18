import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib.pyplot as plt

from thunderfish.dataloader import load_data
from thunderfish.powerspectrum import spectrogram, decibel
from thunderfish.eventdetection import detect_peaks

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

    def detect_calls(self, strict_th=False, det_range=(40000, 150000), plot_debug=False,
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
        calls = get_calls_across_channels(all_recs, plot_spec=True)

        # Compute the Pulse-Intervals:
        diff = np.diff(calls)
        med = np.median(diff)

        # # ToDo: Need to put the diff thing in a new function.
        # inch_factor = 2.54
        # fs = 14
        # fig, ax = plt.subplots(figsize=(50. / inch_factor, 25. / inch_factor))
        #
        # ax.plot(calls[:-1], diff, 'o-', lw=2)
        # ax.plot([calls[0], calls[-1]], [med, med], '--k')
        plt.show()
        quit()

    # Analyze SingleChannel
    elif rec_type == 's':
        
        bat = Batspy(recording, f_resolution=2**9, overlap_frac=.70, dynamic_range=70)  # 2^7 = 128
        bat.compute_spectrogram()
        average_power, peaks, _ = bat.detect_calls(strict_th=True, plot_in_spec=False)

        # Goal now is to create small windows for each call
        # make a time array with the sampling rate
        time = np.arange(0, len(bat.recording_trace) / bat.sampling_rate, 1/bat.sampling_rate)
        window_width = 0.010  # in seconds
        # now follow the call windows
        call_windows = [bat.recording_trace[np.logical_and(time >= bat.t[e]-window_width/2.,
                                                           time <= bat.t[e]+window_width/2.)]
                        for e in peaks]
        call_specs = [[] for e in np.arange(len(call_windows))]

        for i in np.arange(len(call_windows)):
            s, f, t = spectrogram(call_windows[i], samplerate=bat.sampling_rate, fresolution=2**11,
                                  overlap_frac=0.99)

            # set dynamic range
            single_spec_dyn_range = 90
            dec_spec = decibel(s)
            # define maximum; use nanmax, because decibel function may contain NaN values
            ampl_max = np.nanmax(dec_spec)
            dec_spec -= ampl_max + 1e-20  # subtract maximum so that the maximum value is set to lim x--> -0
            dec_spec[dec_spec < -single_spec_dyn_range] = -single_spec_dyn_range
            call_specs[i] = dec_spec
            fig, ax = bat.plot_spectrogram(dec_spec, f, t, ret_fig_and_ax=True)
            import os
            save_path = '../../data/temp_batspy/' + '/'.join(bat.file_path.split('/')[5:-1]) +\
                        '/' + bat.file_name.split('.')[0] + '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            fig.savefig(save_path + 'fig_' + str(i) + '.pdf')
            plt.close(fig)

        print('\nDONE!')