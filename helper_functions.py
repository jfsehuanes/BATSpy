import numpy as np
import matplotlib.pyplot as plt

from thunderfish.eventdetection import detect_peaks

from IPython import embed


def extract_peak_and_th_crossings_from_cumhist(mat, axis, label_array, perc_th=70, neg_sweep_slope=True, plot_debug=False):

    av = np.mean(mat, axis=axis)  # mean over all frequency channels
    abs_av = av - np.min(av)  # make all values positive for the peak-det-algorithm to work
    perc = np.percentile(abs_av, perc_th)

    if axis == 1:  # ToDo: need a cleaner way to solve the artifacts issue in my recordings
        abs_av[np.logical_or((label_array < 98000.), (label_array > 159000.))] = 0.

    thresh = np.min(abs_av)  # threshold for the peak-detector
    if thresh <= 0:  # Fix cases where th <= 0
        thresh = np.mean(abs_av)
    pks, trs = detect_peaks(abs_av, thresh)

    if len(pks) == 0:
        return [], []

    # since more than one peak might be detected, need to choose the one with the highest power
    mx_pk = pks[np.argmax(abs_av[pks])]

    crossings = np.where(np.diff(abs_av > perc))[0]  # gives the crossings where abs_av>perc_th
    # now I extract the sign of crossing differences to the peak. 0 marks the right crossings
    sign_to_pk = np.sign(label_array[crossings] - label_array[mx_pk])
    # look for the crossings pair where the peak is in the middle of both
    call_crossing_idx = np.where(sign_to_pk[:-1] + sign_to_pk[1:] == 0)[0][0]
    call_boundaries = crossings[call_crossing_idx: call_crossing_idx + 2]

    if plot_debug:
        fig, ax = plt.subplots()
        ax.plot(label_array, abs_av)
        ax.plot(label_array[mx_pk], abs_av[mx_pk], 'or', ms=12, mec='k', mew=1.5, alpha=0.7)
        ax.plot([label_array[0], label_array[-1]], [perc, perc], '-k', alpha=0.8)
        ax.plot(label_array[call_boundaries], abs_av[call_boundaries], 'o',
                color='gray', ms=20, mec='k', mew=2, alpha=.7)

    if np.logical_and(axis == 1, neg_sweep_slope):
        return mx_pk, call_boundaries[::-1]
    else:
        return mx_pk, call_boundaries
