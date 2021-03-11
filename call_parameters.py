import os, glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from thunderfish.powerspectrum import decibel
from thunderfish.dataloader import load_data

from IPython import embed


def find_recording(csv_head, start_path ='../../data/phd_data/avisoft_recordings_in_Macaregua/02_recordings_2019/'):

    date = '*' + csv_head.split('/')[0] + '/'
    bat = csv_head.split('/')[1].split('_')[0] + '/'
    recNumber = csv_head.split('/')[1].split('_')[4]

    fullString = start_path + date + bat + 'channel*_recording_0' + recNumber + '.wav'
    recs = glob.glob(fullString)

    if len(recs) != 4:
        raise(ValueError('Something went wrong and I could not find the four channels of the recording you whish '
                         'to analyze'))

    return recs


def best_channel(rec_ls, call_time, window_width=0.008, nfft=2 ** 6):

    peak2NoiseDiff = np.zeros(len(rec_ls))
    specs = []

    for enu, chPath in enumerate(rec_ls):
        print('analyzing ' + chPath + '\n')

        # load current channel
        dat, sr, u = load_data(chPath)
        dat = np.hstack(dat)

        # define the window boundaries
        leftMargin = int(np.floor(call_time*sr - window_width/2*sr))
        rightMargin = int(np.ceil(call_time*sr + window_width/2*sr))
        s, f, t = mlab.specgram(dat[leftMargin:rightMargin], Fs=sr, NFFT=nfft,
                                noverlap=int(0.6 * nfft))  # Compute a high-res spectrogram of the window
        d = decibel(s)
        specs.append(d)

        # compute the peak to noise difference
        noiseInds = np.sum(f > 250000)  # number of indices in frequency above the noise threshold of 250kHz
        noise = np.mean(np.hstack(d[:noiseInds, :]))
        ampPeak = np.max(d[np.logical_and(f >= 100000, f <= 160000)])  # peak within the likely pkfreq range

        peak2NoiseDiff[enu] = noise - ampPeak

        #ToDo: it is a better approach to load all channels with a low resolution spectrogram and use that
        # to decide which channel has the best amplitude.
        #


    embed()
    exit()


if __name__ == '__main__':

    # read the csv
    csv = pd.read_csv('tmp/approach.csv')

    # loop through all sequences in the csv
    for seqInd in np.arange(len(csv)):
        seqName = csv.iloc[:, seqInd].name
        seq = np.array(csv.iloc[:, seqInd])
        seq = seq[~np.isnan(seq)]

        recNames = find_recording(seqName)  # list of strings with paths to recordings

        for enu, call in enumerate(seq):

            # compute a spectrogram of a defined window length
            best_channel(recNames, call)




        # now decide which channel has the strongest amplitude

        embed()
        quit()

    # define the channel with max amplitude and calculate call parameters from it.