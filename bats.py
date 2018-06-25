import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import matplotlib.pyplot as plt

from dataloader import load_data
from powerspectrum import spectrogram, decibel
from peakdetection import detect_peaks

from IPython import embed


class Batspy:

    def __init__(self, file_path, f_resolution=1000., overlap_frac=0.9, dynamic_range=50, pcTape_rec=False):
        self.file_path = file_path
        self.file_name = file_path.split('/')[-1]
        self.freq_resolution = f_resolution  # in Hz
        self.overlap_frac = overlap_frac  # Overlap fraction of the NFFT windows
        self.dynamic_range = dynamic_range  # in dB
        self.pcTape_rec = pcTape_rec  # Was this file recorded by PC-Tape?

        # Flow control booleans
        self.data_loaded = False
        self.spectogram_computed = False
        self.spectogram_plotted = False

    def load_data(self):
        dat, sr, u = load_data(self.file_path)
        self.recording_trace = dat.squeeze()

        if self.pcTape_rec:  # This fixes PC-Tape's bug that writes 1/10 of the samplingrate in the header of the .wav file
            self.sampling_rate = sr * 10.
        else:
            self.sampling_rate = sr

        self.data_loaded = True
        pass

    def compute_spectogram(self, plotit=False):
        if not self.data_loaded:
            self.load_data()
        temp_spec, self.f, self.t = spectrogram(self.recording_trace, self.sampling_rate,
                                                fresolution=self.freq_resolution, overlap_frac=self.overlap_frac)

        # set dynamic range
        dec_spec = decibel(temp_spec)
        ampl_max = np.nanmax(dec_spec)  # define maximum; use nanmax, because decibel function may contain NaN values
        dec_spec -= ampl_max + 1e-20  # subtract maximum so that the maximum value is set to lim x--> -0
        dec_spec[dec_spec < -self.dynamic_range] = -self.dynamic_range

        # Fix NaNs issue
        if True in np.isnan(dec_spec):
            dec_spec[np.isnan(dec_spec)] = - self.dynamic_range

        self.spec_mat = dec_spec

        self.spectogram_computed = True

        if plotit:
            self.plot_spectogram()
        pass

    def plot_spectogram(self, ret_fig_and_ax=False):
        if not self.spectogram_computed:
            self.compute_spectogram()

        inch_factor = 2.54
        fig, ax = plt.subplots(figsize=(56. / inch_factor, 30. / inch_factor))
        im = ax.imshow(self.spec_mat, cmap='jet', extent=[self.t[0], self.t[-1], self.f[0], self.f[-1]], aspect='auto',
                       origin='lower', alpha=0.7)
        # ToDo: Impossible to update the colorbar ticks for them to be multiples of 10!!!
        cb_ticks = np.arange(0, self.dynamic_range + 10, 10)

        cb = fig.colorbar(im)

        cb.set_label('dB')
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        self.spectogram_plotted = True

        if ret_fig_and_ax:
            return fig, ax
        else:
            pass

    def detect_calls(self, det_range=(100000., 180000.), d_range_det_th=0.1, plot_debug=False,
                     plot_in_spec=False, save_spec_w_calls=False):

        if d_range_det_th > 1. or d_range_det_th < 0.:
            raise(ValueError("The detection threshould should be between 0 and 1"))

        # Get the indices of the frequency channels where we want to detect calls
        ind = np.where(np.logical_and(self.f > det_range[0], self.f < det_range[1]))[0]

        # Filter the mean noise out (need to use nanmean() and nanmin(), for there might be NaNs in the spectogram)
        # ToDo: Try a certain percentile instead of the mean! Perhaps I get better results. Visualize mean and percentiles for diff files!
        mean_noise = np.nanmean(self.spec_mat)
        nonoise_spec = self.spec_mat + (self.dynamic_range + mean_noise)

        # Sum over all frequency-channels of interest and set a peak detector
        temp_s = np.sum(nonoise_spec[ind], axis=0) / float(len(nonoise_spec[ind]))
        s = temp_s - np.min(temp_s)
        th = self.dynamic_range * d_range_det_th
        peaks, throughs = detect_peaks(s, th, time=self.t)

        if plot_debug:
            fig, ax = plt.subplots()
            ax.plot(self.t, s)
            ax.plot(peaks, np.ones(len(peaks)) * np.max(s), 'o', ms=20, color='darkred', alpha=.8, mec='k', mew=3)
            ax.plot([self.t[0], self.t[-1]], [th, th], '--k', lw=2.5)
            # plt.show()

        if plot_in_spec:
            spec_fig, spec_ax = self.plot_spectogram(ret_fig_and_ax=True)
            spec_ax.plot(peaks, np.ones(len(peaks))*det_range[1] + 10000, 'o', ms=20, color='darkred',
                         alpha=.8, mec='k', mew=3)
            spec_fig.suptitle(self.file_name.split('.')[0])
            if save_spec_w_calls:
                spec_fig.savefig('test_result/detected_calls/' + self.file_name.split('.')[0] + '.pdf')
        pass


if __name__ == '__main__':

    import glob
    wavefiles = np.sort(glob.glob('../../data/fixed_files/*.wav'))

    for e, wf in enumerate(wavefiles):
        print("\nAnalyzing file %i from %i\n" % (e+1, len(wavefiles)))
        bat = Batspy(wf, dynamic_range=70)
        bat.compute_spectogram()
        bat.detect_calls(plot_debug=True)

        plt.show()
    quit()

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
