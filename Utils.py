import Details as D
import numpy as np
import os
import matplotlib.pyplot as plt
import Maths

def getarrs(num_cond, num_rows, num_epochs=D.numtrialepochs):
    all_avgs = Maths.nans((D.numareas, num_rows, num_cond, num_epochs, D.n_timepoints))
    all_sem = Maths.nans((D.numareas, num_rows, num_cond, num_epochs, D.n_timepoints))
    sigclusters = Maths.nans((D.numareas, num_rows, num_epochs, D.n_timepoints))

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


def savefig(savedir, name, close=True):
    path = f'{D.dir_savefig}{savedir}'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/{name}')
    if close:
        plt.close('all')


def getavgfr(sc, data, cell):
    fr = data.generate_epoch_norm(cell, sc)
    fr_window = np.mean(fr[:, D.rsa_start:D.rsa_stop], axis=1)  # Just take average FR just after the event
    return fr_window
