import numpy as np
import matplotlib.pyplot as plt

from IPython import embed


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


def plot_call_bout_vs_CI(times_to_bout_end, diffs, boutNumber=None):

    inch_factor = 2.54
    fs = 14
    fig, ax = plt.subplots(figsize=(30. / inch_factor, 15. / inch_factor))

    if boutNumber is None:  # i.e. if there was no segregation in bouts
        ax.plot(times_to_bout_end[1:], diffs * 1000., '-o', color='gray', mec='k', ms=10, alpha=.8, lw=1)
            # ax.plot(times_to_bout_end[idx][0], diffs[idx][0] * 1000., 'ok', ms=10, alpha=.5, lw=1)
        ax.set_ylim(-1, 41)
        ax.set_xlabel('Time [s]', fontsize=fs + 4)
    else:
        ax.plot(times_to_bout_end[boutNumber], diffs[boutNumber] * 1000., '-o', color='gray',
                mec='k', ms=10, alpha=.8, lw=1)
        ax.set_xlabel('Time to bout end [s]', fontsize=fs + 4)

    #ax.set_title('Call Intervals during search behavior. %d bouts were detected' % len(diffs))
    ax.set_ylabel('Call Interval [ms]', fontsize=fs+4)
    ax.tick_params(labelsize=fs + 3)
    pass


def save_ipi_sequence(seq2append, behType):

    import os
    import pandas as pd

    valid_types = ['approach', 'attack', 'search']

    if behType not in valid_types:
        raise(ValueError('not a valid behavior type. Valid behaviors are approach, attack and search'))

    # check if there are existing csv-files
    fpath = 'tmp/'+behType+'.csv'
    fileExists = os.path.exists(fpath)

    if fileExists:
        df = pd.read_csv(fpath, header=None)
        df[len(df.columns)] = seq2append

    else:
        df = pd.DataFrame(seq2append)
        pass

    df.to_csv(fpath, header=False, index=False)
    pass


def extract_pulse_sequence(pulse_times, valid_time_range, extra_deletions=[], to_add=[]):

    # focus only on the desired time range
    in_range_pulses = pulse_times[np.logical_and(pulse_times>valid_time_range[0], pulse_times<valid_time_range[1])]

    # delete extra detections or missed detections
    if len(extra_deletions) == 0:
        out = in_range_pulses
        pass
    else:
        del_idx = np.zeros(len(extra_deletions))
        for enu, ed in enumerate(extra_deletions):
            del_idx[enu] = np.argmin(np.abs(in_range_pulses - ed))
        out = np.delete(in_range_pulses, del_idx)

    # add missing pulses
    if len(to_add) > 0:
        out = np.sort(np.hstack((out, to_add)))

    return out