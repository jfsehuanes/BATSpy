import sys
from scipy.io.wavfile import write

from dataloader import load_data

if __name__ == '__main__':

    if len(sys.argv) <= 2:
        print ('ERROR!\nPlease tell me the name of the file you wish to fix as the first argument and the folder '
               'path I should store the fixed file as second argument.')
        quit()
    filename = sys.argv[1]
    if sys.argv[2][-1] == '/':
        out_path = sys.argv[2]
    else:
        out_path = sys.argv[2] + '/'

    out_file = out_path + filename.split('/')[-1]
    out_file = out_file.split('.wav')[0] + '_fix.wav'

    # Load data
    dat, sr, u = load_data(filename)

    # Correct the PC Sampling Rate by multiplying by 10.
    sr *= 10.

    # Save the fixed recording
    write(out_file, int(sr), dat.squeeze())