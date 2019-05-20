import numpy as np
import matplotlib.pyplot as plt
import audioio
from thunderfish.powerspectrum import decibel

from IPython import embed


def gauss_kernel(x, mu, sd):
    return (np.exp(-0.5*(x-mu)**2/(sd**2))) ** 0.01


def gauss_kernel2(x, mu, sd1, sd2):
    y = np.zeros(len(x))
    y[x < mu] = gauss_kernel(x[x < mu], mu, sd1)
    y[x >= mu] = gauss_kernel(x[x >= mu], mu, sd2)

    return y


def generate_single_pulse(pd, pkf=120000., fb=180000., fe=90000., dyn_range=70, samp_freq=600000):

    # calculate the width of the gauss that fits a 2ms call
    f_test = np.arange(-50, 50, 0.01)
    g = decibel(gauss_kernel(f_test, 0, 1))
    th = -dyn_range * 0.5  # db
    g[g < -dyn_range] = -dyn_range

    th_cross = np.where(g >= th)[0]
    f0 = f_test[th_cross[0]]
    f1 = f_test[th_cross[-1]]

    bw = np.abs(fe - fb)
    sd = bw / (f1 - f0)

    sd2 = np.abs((fb - pkf) / f0)
    sd1 = np.abs((fe - pkf) / f1)

    t0 = 0.  # Start time in s
    time = np.arange(t0, pd, 1. / samp_freq)

    # Define the slope of the phase function
    m = (fe - fb) / (pd - t0)

    # define f(t)
    f = m * time + fb

    # Calculate the phase using an integral function and dividing by dt (i.e. the sampling rate)
    phase = np.cumsum(f) / samp_freq

    # Signal
    s = np.sin(2. * np.pi * phase)

    # Apply a gaussian kernel
    s *= gauss_kernel2(f, pkf, sd1, sd2)

    return s


if __name__ == '__main__':

    # define parameters
    durations = np.arange(0.0005, 0.0035, 0.0005)  # pulse durations in s
    intervals = np.arange(0.005, 0.055, 0.005)  # pulse intervals in s
    sampling_rate = 600000  # in Hz
    peak_frequency = 120000  # in Hz
    begin_frequency = 180000  # in Hz
    end_frequency = 90000  # in Hz
    file_duration = 10.  # in s
    dynamic_range = 70  # in db
    folder = 'test_result/'

    for pd in durations:
        s = generate_single_pulse(pd)

        for pi in intervals:
            # create a 10s empty time trace
            signal = np.zeros(int(sampling_rate * file_duration))

            idsCallAndInterval = len(s) + pi*sampling_rate
            max_fitting = np.floor(len(signal) / idsCallAndInterval)
            ids = [int(i * idsCallAndInterval) for i in range(int(max_fitting))]
            for cId in ids:
                signal[cId:int(cId+len(s))] = s

            # save the signal
            out_file = folder + 'pd_' + str(pd*1000) + '__pi_' + str(int(round(pi*1000)))  + '.wav'
            # ToDo: fix filename for pi=5ms (put a 0 before the 5)
            audioio.write_audio(out_file, signal, sampling_rate, 'wav')
    print("\nDONE, stimulation files saved in " + folder)