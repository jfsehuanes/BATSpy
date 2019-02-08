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

    def compute_spectrogram(self, plotit=False):
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
        peaks, throughs = detect_peaks(av_power, th)

        # Get the threshold crossings to determine call duration
        # abv_th, bel_th = threshold_crossings(av_power, th)
        # embed()
        # quit()


        if plot_debug:
            fig, ax = plt.subplots()
            ax.plot(self.t, av_power)
            ax.plot(self.t[peaks], np.ones(len(peaks)) * np.max(av_power), 'o', ms=20, color='darkred', alpha=.8,
                    mec='k', mew=3)
            ax.plot([self.t[0], self.t[-1]], [th, th], '--k', lw=2.5)
            # plt.show()

        if plot_in_spec:
            spec_fig, spec_ax = self.plot_spectrogram(ret_fig_and_ax=True)
            spec_ax.plot(self.t[peaks], np.ones(len(peaks))*det_range[1] + 10000, 'o', ms=20, color='darkred',
                         alpha=.8, mec='k', mew=3)
            spec_fig.suptitle(self.file_name.split('.')[0])
            if save_spec_w_calls:
                spec_fig.savefig('test_result/detected_calls/' + self.file_name.split('.')[0] + '.pdf')
        pass


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
        from multiCH import get_all_ch
        
        all_recs = get_all_ch(recording)

        for rec in all_recs:
            bat = Batspy(rec, f_resolution=2**7, dynamic_range=70)

    elif rec_type == 's':

        # For Myrna's data
        # mouse = Batspy(recording, f_resolution=2**6, overlap_frac=0.4)
        # mouse.compute_spectrogram()
        # mouse.detect_calls(det_range=(30000., 100000.), plot_debug=True, plot_in_spec=True)
        # plt.show()
        # quit()
        
        bat = Batspy(recording, f_resolution=2**8, overlap_frac=.95, dynamic_range=70, pcTape_rec=False)  # 2^7 = 128
        bat.compute_spectrogram()
        bat.detect_calls(plot_in_spec=True)
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
