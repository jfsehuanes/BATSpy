import os
import numpy as np
import matplotlib.pyplot as plt

from thunderfish.eventdetection import detect_peaks

from IPython import embed


def extract_peak_and_th_crossings_from_cumhist(mat, axis, label_array, perc_th=70,
                                               neg_sweep_slope=True, plot_debug=False):

    av = np.mean(mat, axis=axis)  # mean over all frequency channels
    abs_av = av - np.min(av)  # make all values positive for the peak-det-algorithm to work
    perc = np.percentile(abs_av, perc_th)

    # if axis == 1:  # ToDo: need a cleaner way to solve the artifacts issue in my recordings
    #     abs_av[np.logical_or((label_array < 98000.), (label_array > 159000.))] = 0.

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
    try:
        call_crossing_idx = np.where(sign_to_pk[:-1] + sign_to_pk[1:] == 0)[0][0]
    except IndexError:
        embed()
        quit()
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


def save_pis_and_call_parameters(new_pis, callp_dict, save_folder, overwrite=True):

    if save_folder[-1] != '/':
        save_folder += '/'

    # File names
    pis_name = 'all_pulse_intervals.npy'
    cb_name = 'call_begin.npy'
    ce_name = 'call_end.npy'
    fb_name = 'call_fb.npy'
    pf_name = 'call_pk.npy'
    fe_name = 'call_fe.npy'

    # First save the pulse intervals
    if os.path.exists(save_folder + pis_name):
        prev_pis = np.load(save_folder + pis_name)
        np.save(save_folder + pis_name, np.hstack((prev_pis, new_pis)))
    else:
        np.save(save_folder + pis_name, new_pis)

    # Now save the call parameters
    # Call begin
    if os.path.exists(save_folder + cb_name):
        prev_cb = np.load(save_folder + cb_name)
        np.save(save_folder + cb_name, np.hstack((prev_cb, callp_dict['cb'])))
    else:
        np.save(save_folder + cb_name, callp_dict['cb'])

    # Call end
    if os.path.exists(save_folder + ce_name):
        prev_ce = np.load(save_folder + ce_name)
        np.save(save_folder + ce_name, np.hstack((prev_ce, callp_dict['ce'])))
    else:
        np.save(save_folder + ce_name, callp_dict['ce'])

    # Frequency begin
    if os.path.exists(save_folder + fb_name):
        prev_fb = np.load(save_folder + fb_name)
        np.save(save_folder + fb_name, np.hstack((prev_fb, callp_dict['fb'])))
    else:
        np.save(save_folder + fb_name, callp_dict['fb'])

    # Peak Frequency
    if os.path.exists(save_folder + pf_name):
        prev_pf = np.load(save_folder + pf_name)
        np.save(save_folder + pf_name, np.hstack((prev_pf, callp_dict['pf'])))
    else:
        np.save(save_folder + pf_name, callp_dict['pf'])

    # Frequency end
    if os.path.exists(save_folder + fe_name):
        prev_fe = np.load(save_folder + fe_name)
        np.save(save_folder + fe_name, np.hstack((prev_fe, callp_dict['fe'])))
    else:
        np.save(save_folder + fe_name, callp_dict['fe'])

    pass


def specPlotKeyRelease(event):

    global seq_range, to_add, to_del

    fig = plt.gcf()
    print(event.key)
    ix, iy = event.xdata, event.ydata
    if event.key == 'y':
        print('stored x = %.4f in "to_add"' % ix)
        to_add.append(ix)

    elif event.key == 'n':
        print('stored x = %.4f in "to_del"' % ix)
        to_del.append(ix)

    elif event.key == '1':
        print('sequence starts at x = %.4f ' % ix)
        seq_range[0] = ix

    elif event.key == '0':
        print('sequence ends at x = %.4f ' % ix)
        seq_range[1] = ix

    elif event.key == 'enter':
        print('disconnecting plot')
        fig.canvas.mpl_disconnect(cid)

    pass


def manualCallDetectionAdjustment(fig, calls, recording):

    from call_intervals import extract_pulse_sequence, plot_call_bout_vs_CI, save_ipi_sequence
    global seq_range, to_add, to_del

    seq_range = [0, 0]
    to_add = []
    to_del = []

    cid = fig.canvas.mpl_connect('key_release_event', specPlotKeyRelease)
    plt.show()
    
    # The idea is to work with embed and then use the code snippets afterwards while in embed.
    embed()
    quit()

    srch = extract_pulse_sequence(calls, valid_time_range=seq_range,
                                  extra_deletions=to_del, to_add=to_add)
    plot_call_bout_vs_CI(srch, np.diff(srch), boutNumber=None)

    # create the header for the csv
    r = '/'.join(recording.split('/')[-3:])
    shortHeader = '_'.join([r[5:21], 'ch', r[-19], 'rec', r[-6:-4], 'seq_01'])

    save_ipi_sequence(srch, 'glint_machine', shortHeader)
    exit()
