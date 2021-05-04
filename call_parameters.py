import os, glob
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from thunderfish.powerspectrum import decibel
from thunderfish.dataloader import load_data
from thunderfish.eventdetection import threshold_crossings, remove_events

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


def extract_call_boundaries(hist, meanNoise, dbTh=-15):

    dbHist = decibel(hist, None)

    meanNoisedB = decibel(meanNoise, np.max(hist))
    if dbTh - meanNoisedB < 5:  # meanNoisedB has to be at least 5 dB below threshold
        print('+++++++++ noise floor too close to threshold. Ignoring this call +++++++++')
        return dbHist, dbTh, np.nan, np.nan

    up, down = threshold_crossings(dbHist, dbTh)

    up, down = remove_events(up, down, 2)

    if len(up) == 0 or len(down) == 0:
        print('+++++++++ No threshold crossings found! +++++++++')
        return dbHist, dbTh, np.nan, np.nan

    return dbHist, dbTh, up[0]+1, down[-1]  # +1 because threshold_crossings takes the index before th cross


def call_window(dat, sr, callT, winWidth=0.030, pkWidth=0.005, nfft=2 ** 7, overlap_percent=0.6, plotDebug=False,
                dynRange=70):

    # define spec window
    time = np.arange(0, len(dat)/sr, 1/sr)
    windIdx = np.logical_and(time >= callT - winWidth/2., time <= callT + winWidth/2.)

    s, f, t = mlab.specgram(dat[windIdx], Fs=sr, NFFT=nfft,
                            noverlap=int(overlap_percent * nfft))  # Compute a high-res spectrogram of the window

    call_freq_range = (90000, 190000)
    filtered_spec = s[np.logical_and(f > call_freq_range[0], f < call_freq_range[1])]
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
    narrowSpec = filtered_spec[:, np.logical_and(t >= t[pkfTimeIdx]-pkWidth/2, t <= t[pkfTimeIdx]+pkWidth/2)]

    narrowT = np.linspace(t[pkfTimeIdx]-pkWidth/2, t[pkfTimeIdx]+pkWidth/2, np.shape(narrowSpec)[1])

    meanNoise = np.mean(s[f > 250000, :])

    freqHist = np.max(narrowSpec, axis=1)
    timeHist = np.max(narrowSpec, axis=0)

    # RECOMPUTE PEAKFREQ INDEX COORDINATES!!
    # get peak frequency
    peak_f_idx = np.unravel_index(narrowSpec.argmax(),
                                  narrowSpec.shape)

    # define the indices of the peak frequency
    pkfFreqIdx = peak_f_idx[0]
    pkfTimeIdx = peak_f_idx[1]

    missedDetection = False
    dbTimeHist, dbTimeTh, tLeft, tRight = extract_call_boundaries(timeHist, meanNoise, dbTh=-12)

    # now detect the frequency values
    dbFreqHist, dbFreqTh, fRight, fLeft = extract_call_boundaries(freqHist, meanNoise, dbTh=-13)

    if np.any(np.isnan([fRight, fLeft, tRight, tLeft])):
        missedDetection = True

    # debug plot
    if plotDebug:
        inch_factor = 2.54
        fs = 24

        fig = plt.figure(constrained_layout=True, figsize=(56. / inch_factor, 30. / inch_factor))
        gs = fig.add_gridspec(2, 3, height_ratios=(4, 1), width_ratios=(4.85, 4.85, 2))
        ax0 = fig.add_subplot(gs[0, :-1])
        ax1 = fig.add_subplot(gs[1, :-1])
        ax2 = fig.add_subplot(gs[0:-1, -1])

        im = ax0.imshow(decibel(narrowSpec, None), cmap='jet',
                        extent=[narrowT[0], narrowT[-1]+np.diff(narrowT)[0],
                                int(freqs_of_filtspec[0]) / 1000,
                                int(freqs_of_filtspec[-1]+np.diff(freqs_of_filtspec)[0]) / 1000],
                        aspect='auto', origin='lower', alpha=0.7, vmin=-dynRange,
                        vmax=0.)

        ax2.plot(dbFreqHist, freqs_of_filtspec/1000, 'k', lw=2)
        ax2.plot(np.ones(len(freqs_of_filtspec))*dbFreqTh, freqs_of_filtspec/1000, '--b', lw=2)

        # now plot the detected parameters inside the spectrogram
        if not missedDetection:
            tBegin = narrowT[tLeft]
            tEnd = narrowT[tRight]
            fBegin = freqs_of_filtspec[fLeft]
            fEnd = freqs_of_filtspec[fRight]
            ax0.plot([tBegin, tEnd], np.array([fBegin, fEnd]) / 1000, 'ow', ms=10, mec='k', mew=2)

        ax0.plot(narrowT[peak_f_idx[1]], freqs_of_filtspec[peak_f_idx[0]] / 1000,
                 'or', ms=13, mec='k', mew=2)

        ax0.set_ylabel('Frequency [kHz]', fontsize=fs + 1)

        ax1.set_xlabel('Time [sec]', fontsize=fs + 1)
        # ax0.set_xlabel('Time [sec]', fontsize=fs + 1)

        for c_ax in [ax0, ax1, ax2]:
            c_ax.tick_params(labelsize=fs)


        # Plot the soundwave underneath the spectrogram
        ax1.plot(narrowT, dbTimeHist, 'k', lw=2)
        ax1.plot(narrowT, np.ones(len(narrowT)) * dbTimeTh, '--b', lw=2)

        # Share the time axis of spectrogram and raw sound trace
        ax1.get_shared_x_axes().join(ax0, ax1)

        # Remove time xticks of the spectrogram
        ax0.xaxis.set_major_locator(plt.NullLocator())

    if missedDetection:
        print('+++++++++++++++No boundaries found. inserting nans!!+++++++++++++')
        return np.nan, np.nan, np.nan, np.nan
    else:
        if not plotDebug:
            fBegin = freqs_of_filtspec[fLeft]
            fEnd = freqs_of_filtspec[fRight]
            tBegin = narrowT[tLeft]
            tEnd = narrowT[tRight]
        cDur = tEnd - tBegin
        pkFreq = freqs_of_filtspec[pkfFreqIdx]
        return cDur, fBegin, fEnd, pkFreq


if __name__ == '__main__':

    # read the csv
    csv = pd.read_csv('tmp/approach.csv')
    path = '../../data/phd_data/tmp_call_figures/approach/'

    # loop through all sequences in the csv
    for seqInd in range(csv.shape[1]):
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
        callsMask = np.zeros(len(calls))
        enu = 0
        for channel in set(bch):
            # load data
            dat, sr, u = load_data(recNames[channel])
            dat = np.hstack(dat)

            # ToDo: need to find a way to restructure the calls. They're being analyzed shuffled
            # Create a csv file with the sequence name as title and then the columns time, bch, call duration,
            # fBeg, fEnd, and pkfreq

            # embed()
            # quit()

            for callT in calls[bch == channel]:

                print('analyzing call %i' % (enu+1))  # calls are not analyzed in order ;)

                # compute a high res spectrogram of a defined window length
                dur, fb, fe, pf = call_window(dat, sr, callT, plotDebug=True)

                # save the debug figure
                fig = plt.gcf()
                fig.suptitle(seqName + '__CALL#' + '{:03}'.format(enu + 1), fontsize=14)
                fig.savefig(path + 'plots/' + '__'.join(seqName.split('/')) + '__CALL#'+'{:03}'.format(enu+1)+'.pdf')
                plt.close(fig)

                # save the parameters
                callDur[enu] = dur
                freqBeg[enu] = fb
                freqEnd[enu] = fe
                peakFreq[enu] = pf
                callsMask[enu] = callT

                enu += 1

        # Reorder the arrays and create a csv
        sortedInxs = np.argsort(callsMask)
        paramsdf = pd.DataFrame({'callTime': callsMask[sortedInxs], 'bch': bch[sortedInxs],
                                 'callDur': callDur[sortedInxs], 'fBeg': freqBeg[sortedInxs],
                                 'fEnd': freqEnd[sortedInxs], 'pkfreq': peakFreq[sortedInxs]})
        paramsdf.to_csv(path_or_buf=path + 'csvs/' + '__'.join(seqName.split('/')) + '.csv', index=False)
        

        # ToDo: Find a way to optimize the code using parallel computing.
        # ToDo: Make this also compatible with batspy

    print('FINISSSEEEDD')
    quit()
