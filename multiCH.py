# This script analyzes files coming from Multi-Channel recordings

import numpy as np

def get_all_ch(single_filename):
    import glob
    path = '/'.join(single_filename.split('/')[:-1])
    f = single_filename.split('/')[-1]
    all_recs = f.split('_')[0][:-1] + '*_' + '_'.join(f.split('_')[1:])
    ch_list = glob.glob('/'.join([path, all_recs]))

    return np.sort(ch_list)
