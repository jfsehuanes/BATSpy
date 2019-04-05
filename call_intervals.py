import numpy as np
import matplotlib.pyplot as plt


def get_CI_and_call_bouts(call_times, bout_threshold_factor=3):

    """

    Parameters
    ----------
    call_times: array or list.
    Array with the times of the calls
    bout_threshold_factor: int.
    factor by which the meadian(diff(calls)) is multiplied in order to set a threshold that segregates calls into bouts.

    Returns
    -------
    call_times_2end: array.
    array with the call times centered to the end of the bout, i.e. the last element is equal to 0
    bout_diffs: array.
    The corresponding Call intervals to call_times_2end. Note that each call interval corresponds to the interval
    between current and next call.

    """
    diff = np.diff(call_times)

    bout_th = np.median(diff) * bout_threshold_factor
    bout_ids = np.where(diff > bout_th)[0]
    bout_ranges = [[bout_ids[e], bout_ids[e + 1]] for e in
                   np.arange(len(bout_ids) - 1)]  # bout ranges without boundaries

    bout_ranges = np.vstack(([0, bout_ids[0]], bout_ranges, [bout_ids[-1], len(diff)]))  # boundaries inserted

    call_times_2end = []
    bout_diffs = []
    for cb in bout_ranges:
        if len(diff[cb[0]: cb[1]]) > 10:
            call_times_2end.append(call_times[cb[0]: cb[1]][1:] - call_times[cb[0]: cb[1]][-1])
            bout_diffs.append(diff[cb[0]: cb[1]][1:])

    return np.array(call_times_2end), np.array(bout_diffs)


def plot_call_bout_vs_CI(times_to_bout_end, diffs):

    inch_factor = 2.54
    fs = 14
    fig, ax = plt.subplots(figsize=(30. / inch_factor, 15. / inch_factor))

    for idx in np.arange(len(diffs)):
        ax.plot(times_to_bout_end[idx], diffs[idx] * 1000., '-k', alpha=.3, lw=1)
        ax.plot(times_to_bout_end[idx][0], diffs[idx][0] * 1000., 'ok', ms=10, alpha=.5, lw=1)

    ax.set_title('Call Intervals during search behavior. %d bouts were detected' % len(diffs))
    ax.set_xlabel('Time to bout end [s]', fontsize=fs)
    ax.set_ylabel('Call Interval [ms]', fontsize=fs)
    pass
