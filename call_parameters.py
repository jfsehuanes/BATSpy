import os, glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from thunderfish.powerspectrum import decibel
from thunderfish.dataloader import load_data
from thunderfish.eventdetection import detect_peaks

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


def best_channel(rec_ls, calls, window_width=0.008, nfft=2 ** 8, overlap_percent=0.8, thresholdTolerance=20.):

    peak2NoiseDiff = [[] for e in np.arange(len(rec_ls))]

    for enu, chPath in enumerate(rec_ls):
        print('analyzing ' + chPath.split('/')[-1].split('_')[0] + '...')

        # load current channel
        dat, sr, u = load_data(chPath)
        dat = np.hstack(dat)

        s, f, t = mlab.specgram(dat, Fs=sr, NFFT=nfft, noverlap=int(overlap_percent * nfft))  # med resolution specgram
        d = decibel(s)

        timeBools = [np.logical_and(t >= c - window_width / 2, t <= c + window_width / 2) for c in calls]
        freqBools = np.logical_and(f >= 100000, f <= 160000)  # peak within the likely pkfreq range

        callMaxAmps = np.array([np.max(d[np.ix_(freqBools, timeBools[a])]) for a in np.arange(len(timeBools))])

        # compute the peak to noise difference
        noiseInds = np.sum(f > 250000)  # number of indices in frequency above the noise threshold of 250kHz
        noise = np.mean(np.hstack(d[:noiseInds, :]))

        # need to solve the cases where inf is present.
        p2nd = np.abs(noise - callMaxAmps)
        if np.isinf(p2nd).any():
            print('WARNING! found infs!!')
            p2nd = np.nan_to_num(p2nd, posinf=0., neginf=0.)  # sets infs in array to 0
        peak2NoiseDiff[enu] = p2nd

    loudCh = np.argmax(peak2NoiseDiff, axis=0)
    validCallsIds = np.where(np.max(peak2NoiseDiff, axis=0) >= thresholdTolerance)[0]

    retCalls = calls[validCallsIds]
    retChs = loudCh[validCallsIds]

    return retCalls, retChs


def call_window(recFile, callT, winWidth=0.008, nfft=2 ** 10, overlap_percent=0.8, plotDebug=False, dynRange=70):
    # load data
    dat, sr, u = load_data(recFile)
    dat = np.hstack(dat)

    # define spec window
    time = np.arange(0, len(dat)/sr, 1/sr)
    windIdx = np.logical_and(time >= callT - winWidth/2., time <= callT + winWidth/2.)

    s, f, t = mlab.specgram(dat[windIdx], Fs=sr, NFFT=nfft,
                            noverlap=int(overlap_percent * nfft))  # Compute a high-res spectrogram of the window

    dec_spec = decibel(s)

    call_freq_range = (50000, 180000)
    filtered_spec = dec_spec[np.logical_and(f > call_freq_range[0], f < call_freq_range[1])]
    freqs_of_filtspec = np.linspace(call_freq_range[0], call_freq_range[-1], np.shape(filtered_spec)[0])

    # measure noise floor
    noiseInds = np.sum(f > 250000)  # number of indices in frequency above the noise threshold of 250kHz
    meanNoise = np.mean(np.hstack(dec_spec[:noiseInds, :]))
    noise_floor = np.min(np.hstack(dec_spec[:noiseInds, :]))

    lowest_decibel = noise_floor

    # get peak frequency
    peak_f_idx = np.unravel_index(filtered_spec.argmax(),
                                  filtered_spec.shape)

    left_from_peak = np.arange(peak_f_idx[1] - 1, -1, -1, dtype=int)
    right_from_pk = np.arange(peak_f_idx[1] + 1, len(t), dtype=int)

    mainHarmonicTrace = []  # idx array that marks the trace using a coordinate system in the specgram-matrix
    db_th = 7.0
    f_tol_th = 20000  # in Hz
    t_tol_th = 0.0005  # in s

    freq_tolerance = np.where(np.cumsum(np.diff(freqs_of_filtspec)) > f_tol_th)[0][0]
    time_tolerance = np.where(np.cumsum(np.diff(t)) > t_tol_th)[0][0]

    # first start from peak to right
    f_ref = peak_f_idx[0]
    t_ref = peak_f_idx[1]
    mainHarmonicTrace.append([peak_f_idx[0], peak_f_idx[1]])
    for ri in right_from_pk:
        pi, _ = detect_peaks(filtered_spec[:, ri], db_th)
        pi = pi[filtered_spec[pi, ri] > meanNoise]

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
        pi = pi[filtered_spec[pi, li] > meanNoise]

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

    # now save the parameters!
    callEnd = t[mainHarmonicTrace[-1][1]]
    callBeg = t[mainHarmonicTrace[0][1]]
    callDur = callEnd - callBeg

    freqBeg = freqs_of_filtspec[mainHarmonicTrace[0][0]]
    freqEnd = freqs_of_filtspec[mainHarmonicTrace[-1][0]]
    peakFreq = freqs_of_filtspec[peak_f_idx[0]]

    # debug plot
    if plotDebug:
        inch_factor = 2.54
        fs = 24

        fig = plt.figure(constrained_layout=True, figsize=(56. / inch_factor, 30. / inch_factor))
        gs = fig.add_gridspec(2, 3, height_ratios=(4, 1), width_ratios=(4.85, 4.85, .3))
        ax0 = fig.add_subplot(gs[0, :-1])
        ax1 = fig.add_subplot(gs[1, :-1])
        ax2 = fig.add_subplot(gs[0:-1, -1])

        # need to create a time array that fits with the time of the call
        plotT = np.linspace(time[windIdx][0], time[windIdx][-1], len(t))

        # define max and min of the spectrogram to be plotted
        maxSpec = np.max(filtered_spec)
        minSpec = np.min(filtered_spec - maxSpec)

        im = ax0.imshow(filtered_spec - maxSpec, cmap='jet',
                        extent=[plotT[0], plotT[-1],
                                int(freqs_of_filtspec[0]) / 1000,
                                int(freqs_of_filtspec[-1]) / 1000],  # divide by 1000 for kHz
                        aspect='auto', interpolation='hanning', origin='lower', alpha=0.7, vmin=minSpec - minSpec/4,
                        vmax=0.)

        cb = fig.colorbar(im, cax=ax2)

        # now plot the detected parameters inside the spectrogram
        ax0.plot(plotT[mainHarmonicTrace[:, 1]], freqs_of_filtspec[mainHarmonicTrace[:, 0]] / 1000,
                 'ow', ms=10, mec='k', mew=2)
        ax0.plot(plotT[peak_f_idx[1]], freqs_of_filtspec[peak_f_idx[0]] / 1000,
                 'or', ms=13, mec='k', mew=2)

        cb.set_label('dB', fontsize=fs)
        ax0.set_ylabel('Frequency [kHz]', fontsize=fs + 1)

        ax1.set_ylabel('Amplitude [a.u.]', fontsize=fs + 1)
        ax1.set_xlabel('Time [sec]', fontsize=fs + 1)
        ax0.set_xlabel('Time [sec]', fontsize=fs + 1)

        for c_ax in [ax0, ax1, ax2]:
            c_ax.tick_params(labelsize=fs)

        ax0.set_ylim(int(freqs_of_filtspec[0]) / 1000, int(freqs_of_filtspec[-1]) / 1000)

        # Plot the soundwave underneath the spectrogram
        ax1.set_facecolor('black')
        ax1.plot(time[windIdx], dat[windIdx], color='yellow', lw=2, rasterized=True)

        # Share the time axis of spectrogram and raw sound trace
        ax1.get_shared_x_axes().join(ax0, ax1)
        ax0.set_xlim(time[windIdx][0], time[windIdx][-1])

        # Remove time xticks of the spectrogram
        ax0.xaxis.set_major_locator(plt.NullLocator())

    return callDur, freqBeg, freqEnd, peakFreq


if __name__ == '__main__':

    # read the csv
    csv = pd.read_csv('tmp/approach.csv')

    # loop through all sequences in the csv
    for seqInd in np.arange(len(csv)):
        seqName = csv.iloc[:, seqInd].name
        print("\ncurrent sequence: %s" % seqName)
        seq = np.array(csv.iloc[:, seqInd])
        seq = seq[~np.isnan(seq)]

        recNames = find_recording(seqName)  # list of strings with paths to recordings

        # Now load all channels with a low resolution spectrogram and define which channel has the highest amplitude
        calls, bch = best_channel(recNames, seq)
        print('\n%i calls detected above threshold, proceeding with parameter extraction...' % len(calls))

        callDur = np.zeros(len(calls))
        freqBeg = np.zeros(len(calls))
        freqEnd = np.zeros(len(calls))
        peakFreq = np.zeros(len(calls))

        for enu, callT in enumerate(calls):
            # compute a high res spectrogram of a defined window length
            dur, fb, fe, pf = call_window(recNames[bch[enu]], callT, plotDebug=True)

            callDur[enu] = dur
            freqBeg[enu] = fb
            freqEnd[enu] = fe
            peakFreq[enu] = pf

        embed()
        quit()
