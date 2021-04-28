import os, glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from thunderfish.powerspectrum import decibel
from thunderfish.dataloader import load_data
from thunderfish.eventdetection import detect_peaks, threshold_crossings, minmax_threshold, hist_threshold,\
    percentile_threshold

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


def best_channel(rec_ls, calls, window_width=0.010, nfft=2 ** 8, overlap_percent=0.8, thresholdTolerance=20.):

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


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def extract_freqs():

    pass


def extract_call_boundaries(summed_hist, peakInx, histAxis, threshType=None, convBoxPoints=None, thFac=0.4):

    if convBoxPoints == None:
        convBoxPoints = int(np.ceil(0.15 * len(summed_hist)))
    # Create a convolution filter for smoothing out the signal and so come around the Nullstellen-issue
    smthHist = smooth(summed_hist, convBoxPoints)

    # find the crossings of the smoothed summed histogram
    if threshType == 'minmax':
        th = minmax_threshold(smthHist, thresh_fac=thFac)
    elif threshType == 'hist':
        th, _ = hist_threshold(smthHist, thresh_fac=thFac)
    else:
        raise(TypeError("not a valid threshold type. Please specify between 'minmax' and 'hist'"))

    up, down = threshold_crossings(smthHist, th)

    if th >= np.max(smthHist):
        print('+++++++++ WARNING!! Call boundaries not detected! +++++++++')
        return smthHist, th, np.nan, np.nan

    elif len(up) == 0 or len(down) == 0:
        print('increasing threshold!')
        extract_call_boundaries(summed_hist, peakInx, histAxis, threshType='minmax', convBoxPoints=convBoxPoints,
                                thFac=thFac*2)

    # check whether the peak time index lies within the detected time call boundaries
    peakInTheMiddle = False
    crossInxUp = np.where(peakInx - up > 0)[0][-1]
    crossInxDown = np.where(down - peakInx > 0)[0][0]

    if crossInxUp == crossInxDown:
        peakInTheMiddle = True

    if peakInTheMiddle:
        return smthHist, th, up[crossInxUp], down[crossInxDown]
    else:
        print('+++++++++ WARNING!! Call peak not within call boundaries! +++++++++')
        return smthHist, th, np.nan, np.nan


def call_window(recFile, callT, winWidth=0.030, nfft=2 ** 9, overlap_percent=0.6, plotDebug=False, dynRange=70):
    # load data
    dat, sr, u = load_data(recFile)
    dat = np.hstack(dat)

    # define spec window
    time = np.arange(0, len(dat)/sr, 1/sr)
    windIdx = np.logical_and(time >= callT - winWidth/2., time <= callT + winWidth/2.)

    s, f, t = mlab.specgram(dat[windIdx], Fs=sr, NFFT=nfft,
                            noverlap=int(overlap_percent * nfft))  # Compute a high-res spectrogram of the window

    dec_spec = decibel(s)

    call_freq_range = (50000, 200000)
    filtered_spec = dec_spec[np.logical_and(f > call_freq_range[0], f < call_freq_range[1])]
    freqs_of_filtspec = np.linspace(call_freq_range[0], call_freq_range[-1], np.shape(filtered_spec)[0])

    # get peak frequency
    peak_f_idx = np.unravel_index(filtered_spec.argmax(),
                                  filtered_spec.shape)

    # first start from peak to right
    pkfFreqIdx = peak_f_idx[0]
    pkfTimeIdx = peak_f_idx[1]

    # need to create a time array that fits with the time of the call
    newT = np.linspace(time[windIdx][0], time[windIdx][-1], len(t))

    # call begin and end computed using summed histograms along both the time and frequency axes of the spectrogram
    narrowSpec = s[np.logical_and(f > call_freq_range[0], f < call_freq_range[1])]
    # meanNoise = np.mean(narrowSpec[:noiseInds, :])
    freqHist = np.sum(narrowSpec, axis=1)
    timeHist = np.sum(narrowSpec, axis=0)

    # ToDo: I think it's better to work with a hist_threshold for the freqs and minmax_threshold for time
    # smthFreqHist, fBegin, fEnd = extract_call_boundaries(freqHist, pkfFreqIdx, freqs_of_filtspec, threshType='minmax',
    #                                                      thFac=0.4)

    try:
        smthTimeHist, smthTHistTh, tLeft, tRight = extract_call_boundaries(timeHist, pkfTimeIdx, newT,
                                                                       threshType='minmax', thFac=0.4)
    except:
        embed()
        exit()

    # now detect the frequency values
    smthFreqHist, smthFHistTh, fRight, fLeft = extract_call_boundaries(freqHist, pkfFreqIdx, freqs_of_filtspec,
                                                                       threshType='minmax', thFac=0.4)
    # embed()
    # exit()

    fBegin = freqs_of_filtspec[fLeft]
    fEnd = freqs_of_filtspec[fRight]

    tBegin = newT[tLeft]
    tEnd = newT[tRight]
    cDur = tEnd - tBegin
    pkFreq = freqs_of_filtspec[pkfFreqIdx]

    # debug plot
    # ToDo: add the summed histograms to the DeBug plot
    if plotDebug:
        inch_factor = 2.54
        fs = 24

        fig = plt.figure(constrained_layout=True, figsize=(56. / inch_factor, 30. / inch_factor))
        gs = fig.add_gridspec(2, 3, height_ratios=(4, 1), width_ratios=(4.85, 4.85, .3))
        ax0 = fig.add_subplot(gs[0, :-1])
        ax1 = fig.add_subplot(gs[1, :-1])
        ax2 = fig.add_subplot(gs[0:-1, -1])

        # define max and min of the spectrogram to be plotted
        maxSpec = np.max(filtered_spec)
        minSpec = np.min(filtered_spec - maxSpec)

        im = ax0.imshow(filtered_spec - maxSpec, cmap='jet',
                        extent=[newT[0], newT[-1],
                                int(freqs_of_filtspec[0]) / 1000,
                                int(freqs_of_filtspec[-1]) / 1000],  # divide by 1000 for kHz
                        aspect='auto', interpolation='hanning', origin='lower', alpha=0.7, vmin=minSpec - minSpec/4,
                        vmax=0.)

        ax2.plot(smthFreqHist, freqs_of_filtspec/1000, 'k', lw=2)
        ax2.plot(np.ones(len(freqs_of_filtspec))*smthFHistTh, freqs_of_filtspec/1000, '--b', lw=2)

        # now plot the detected parameters inside the spectrogram
        ax0.plot([tBegin, tEnd], np.array([fBegin, fEnd]) / 1000,
                 'ow', ms=10, mec='k', mew=2)
        ax0.plot(newT[peak_f_idx[1]], freqs_of_filtspec[peak_f_idx[0]] / 1000,
                 'or', ms=13, mec='k', mew=2)

        ax0.set_ylabel('Frequency [kHz]', fontsize=fs + 1)

        ax1.set_xlabel('Time [sec]', fontsize=fs + 1)
        # ax0.set_xlabel('Time [sec]', fontsize=fs + 1)

        for c_ax in [ax0, ax1, ax2]:
            c_ax.tick_params(labelsize=fs)

        ax0.set_ylim(int(freqs_of_filtspec[0]) / 1000, int(freqs_of_filtspec[-1]) / 1000)

        # Plot the soundwave underneath the spectrogram
        ax1.plot(newT, smthTimeHist, 'k', lw=2)
        ax1.plot(newT, np.ones(len(newT)) * smthTHistTh, '--b', lw=2)

        # Share the time axis of spectrogram and raw sound trace
        ax1.get_shared_x_axes().join(ax0, ax1)
        ax0.set_xlim(time[windIdx][0], time[windIdx][-1])

        # Remove time xticks of the spectrogram
        ax0.xaxis.set_major_locator(plt.NullLocator())

    return cDur, fBegin, fEnd, pkFreq


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
            print("Call %i of %i..." % (enu+1, len(calls)))
            # compute a high res spectrogram of a defined window length
            dur, fb, fe, pf = call_window(recNames[bch[enu]], callT, plotDebug=True)

            callDur[enu] = dur
            freqBeg[enu] = fb
            freqEnd[enu] = fe
            peakFreq[enu] = pf

        embed()
        quit()
