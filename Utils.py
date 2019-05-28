import Details as D
import numpy as np
import os
import matplotlib.pyplot as plt


def getarrs(num_cond, num_rows):
    all_avgs = np.zeros((D.numareas, num_rows, num_cond, D.numtrialepochs, D.num_timepoints))
    all_sem = np.zeros((D.numareas, num_rows, num_cond, D.numtrialepochs, D.num_timepoints))
    sigclusters = np.zeros((D.numareas, num_rows, D.numtrialepochs, D.num_timepoints))

    # Fill with nans
    all_avgs.fill(np.nan)
    all_sem.fill(np.nan)
    sigclusters.fill(np.nan)

    return all_avgs, all_sem, sigclusters


def updatecounts(counter, cell, n):
    counter[cell] = 1
    if int(sum(counter)) % 20 == 0:
        print(f'{int(sum(counter))}/{n}')


def normalisedata(arr):
    if D.rsa_norm_method == 'normalise':
        arr -= np.min(arr)
        arr /= np.max(arr)
    elif D.rsa_norm_method == 'standardise':
        arr -= np.mean(arr)
        arr /= np.std(arr)
    else:
        raise Exception('Error!')
    return arr


def savefig(savedir, name):
    path = f'{D.dir_savefig}{savedir}'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/{name}')
    plt.close('all')


def getavgfr(sc, data, cell):
    fr = data.generatenormalisedepoch(cell, sc)
    fr_window = np.mean(fr[:, D.rsa_start:D.rsa_stop], axis=1)  # Just take average FR just after the event
    return fr_window
